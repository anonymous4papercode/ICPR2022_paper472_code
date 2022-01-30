from sqlalchemy import false
import torchvision.models as models
from torch.nn import Parameter

from util import *
import torch
import torch.nn as nn

from functools import partial
from einops.layers.torch import Rearrange, Reduce

import timm

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=2, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


class mixer(nn.Module):
    def __init__(self, mix_depth, patches, hidden_dim=256, dropout=0.):
        super(mixer, self).__init__()
        self.patches = patches
        self.dim = hidden_dim
        self.depth = mix_depth
        
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.layers = nn.Sequential(
            Rearrange('(h p) w -> h p w', p=self.patches),
            *[nn.Sequential(
                PreNormResidual(self.dim, FeedForward(self.patches, dense=chan_first, dropout=dropout)),
                PreNormResidual(self.dim, FeedForward(self.dim, dense=chan_last, dropout=dropout)),
            ) for _ in range(self.depth)],
            Rearrange('h p w -> (h p) w', p=self.patches)
        )
    
    def forward(self, feat):
        for layer in self.layers:
            feat = layer(feat)
        return feat


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class CSRAPooling(nn.Module):
    def __init__(self,hidden_dim):
        super(CSRAPooling, self).__init__()
        self.fc=nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0)
        self.avg_pooling = nn.AvgPool2d(14, 14)
        self.max_pooling = nn.MaxPool2d(14, 14)
        self.Lambda = 0.1
    def forward(self,feature):
        feature=self.fc(feature)
        avg = self.avg_pooling(feature);
        max = self.max_pooling(feature);
        return avg + self.Lambda*max;

class MultiHeadCSRAPooling(nn.Module):
    def __init__(self, hidden_dim, heads=1):
        super(MultiHeadCSRAPooling,self).__init__()
        self.layers=nn.Sequential(*[CSRAPooling(hidden_dim=hidden_dim) for i in range(heads)])
    def forward(self, feature):
        sum=0
        for layer in self.layers:
            sum+=layer(feature)
        return sum/len(self.layers)
        


class MixResnet(nn.Module):
    def __init__(self, model, num_classes, base_patches=1, mix_layers=2, in_channel=300, t=0,freeze=0, adj_file=None):
        super(MixResnet, self).__init__()
        self.features = model
        if freeze:
            # 预训练模型resnet101不更新参数
            for name, param in self.features.named_parameters():
                print(f'{name}层不更新参数')
                param.requires_grad = False
        
        self.num_classes = num_classes
        
        hidden_dim = 2432

        self.pooling = MultiHeadCSRAPooling(heads=8,hidden_dim=hidden_dim)
        
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.gc1 = GraphConvolution(in_channel, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)

        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.mixers = nn.Sequential(
            *[mixer(mix_depth=2, patches=base_patches*(2**(i+1)), hidden_dim=in_channel,dropout=0.05*(mix_layers-i)) for i in range(mix_layers)])

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        inp = inp[0]
        
        x = self.mixers(inp)
        
        adj = gen_adj(self.A).detach()

        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp, freeze):
        res = [
                {'params': self.mixers.parameters(), 'lr': lr},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]
        if not freeze:
            res.append({'params': self.features.parameters(), 'lr': lr * lrp})
        return res


def mix_resnet101(num_classes, t, base_patches=1, mix_layers=2, pretrained=False,freeze=0, adj_file=None, in_channel=300, use_resnet101=False):
    if use_resnet101:
        model = models.resnet101(pretrained=pretrained)
    else:
        model = timm.create_model("tresnet_l_448",pretrained=pretrained)
        model = model.body
    return MixResnet(model, num_classes,base_patches=base_patches,mix_layers=mix_layers, t=t,freeze=freeze, adj_file=adj_file, in_channel=in_channel)
