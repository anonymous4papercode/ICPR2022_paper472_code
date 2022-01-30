from numpy import mod
import torch
import torch.nn as nn
from copy import deepcopy
import gc   
class PPIR(nn.Module):
    @torch.no_grad()
    def __init__(self, model):
        super(PPIR, self).__init__()
        model = deepcopy(model)
        self.prev_opt_thetas = list(model.parameters())
        self.lamda = 0.5
        self.epsilon = 0.1
        self.W = {}
        self.omega = {}
        self.p_old = {}
        for i in range(len(self.prev_opt_thetas)):
            theta = self.prev_opt_thetas[i]
            self.W[i] = theta.data.clone().zero_()
            self.p_old[i] = theta.data.clone()
            self.omega[i] = theta.detach().clone().zero_()

        self.update_omega(model)
        del model
        gc.collect()


    @torch.no_grad() 
    def update_omega(self, model):        
        model = deepcopy(model)
        cur_thetas = list(model.parameters())
        for i in range(len(cur_thetas)):
            cur_theta = cur_thetas[i]
            if cur_theta.grad is not None:
                self.W[i].add_(-cur_theta.grad * (cur_theta.detach() - self.p_old[i]))
                self.p_old[i] = cur_theta.detach().clone()
            prev_opt_theta = self.prev_opt_thetas[i]
            self.prev_opt_thetas[i] = cur_theta
            p_change = cur_theta - prev_opt_theta
            omega_new = self.W[i] / (p_change ** 2 + self.epsilon)
            self.omega[i] += omega_new
        
        del model
        gc.collect()
    
    @torch.no_grad()
    def forward(self, model):
        # with torch.no_grad():
        loss = 0
        model = deepcopy(model)
        cur_thetas = list(model.parameters())
        for i in range(len(cur_thetas)):
            cur_theta = cur_thetas[i]
            prev_opt_theta = self.prev_opt_thetas[i]
            loss += self.lamda / 2 * torch.sum(self.omega[i] * (cur_theta - prev_opt_theta) ** 2)

        del model
        gc.collect()
        return loss
