# import torch
from os import stat
import re
import torch.utils.data
import argparse
from engine import *
from models import *
from voc import *
from loss import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=30, type=int,
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--patches', '-pt', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--t', '-t', default=0.4, type=float,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--mix-layers', '-mls', default=2, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--pretrained', '-pre', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--warm-up', '-wu', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--ppir', '-pp', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--freeze', '-f', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    args.model = 'mix'
    key = "runs/voc2007/" + args.model
    writer = SummaryWriter(key)

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = Voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
    val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')

    num_classes = 20

    t = args.t
    # load model
    
    model = mix_resnet101(num_classes=num_classes, pretrained=args.pretrained,freeze=args.freeze,  base_patches=args.patches, mix_layers=args.mix_layers, t=t, adj_file='data/voc/voc_adj.pkl')


    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp, args.freeze),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/voc2007/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['model_name'] = args.model
    state['writer'] = writer
    state['use_PPIR'] = args.ppir
    state['warm_up'] = args.warm_up
    state['dataset'] = 'voc'
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    main_voc2007()
