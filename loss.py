import torch
import torch.nn as nn
from loss_utils.vgg import Vgg16
from utils import log
from torchvision import transforms
import numpy as np
from loss_utils import pytorch_msssim as torch_msssim


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()

        #self.vgg = Vgg16(requires_grad=False)
        #self.mse_vgg = nn.MSELoss()
        #self.mse = nn.MSELoss()
        
    def loss_mix_v3(self, y_true, y_pred):
    
        # weights
        alpha = 0.2
        l1_w = 1-alpha
        msssim_w = alpha

        l1_value = torch.mean(torch.abs(y_pred - y_true) * torch.abs(y_true - 0.5))
        msssim_value = torch.mean(1-torch_msssim.msssim(y_pred, y_true)) # must be (0,1) rangee

        return (msssim_w*msssim_value) + (l1_w*l1_value)

    def forward(self, x, y):
        
        # feature loss
        #x_vgg = self.vgg(x).relu2_2
        #y_vgg = self.vgg(y).relu2_2
        
        # mix loss v3 + feature loss
        loss = self.loss_mix_v3(y, x)# * 0.8 + self.mse_vgg(y_vgg, x_vgg) * 0.2

        return loss
