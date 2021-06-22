import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_c(batch_size,M,b,device='cuda'):
    """
    Convert uniform random variable to Laplace random variable
    
    Inputs:
        - batch_size:   batch size training run
        - M:            Number of transport operator dictionary elements
        - zeta:         Spread parameter for Laplace distribution
    
    Outputs:
        - c:            Vector of sampled Laplace random variables [batch_sizexM]
    """
 
    u = (torch.rand(batch_size,M)-0.5).to(device)
    c = -torch.mul(torch.mul(torch.sign(u),torch.log((1.0-2.0*torch.abs(u)).clamp(min=1e-10))),b)
    return c
