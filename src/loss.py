import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.modules.loss as loss
import numpy as np
import torch.fft as fft


def loss_fn(loss_name):
    if loss_name == 'MSE':
        return loss.MSELoss()
    else:
        raise NotImplementedError('Loss function not implemented')
    
    
    
    


