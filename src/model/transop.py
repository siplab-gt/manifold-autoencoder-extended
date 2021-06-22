import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransOp_expm(nn.Module):

    def __init__(self, M=6, N=3, var=0.1):
        super(TransOp_expm, self).__init__()
        self.psi = nn.Parameter(torch.mul(torch.randn((M, N, N)), var), requires_grad=True)
        self.psi.data = self.psi.data / self.psi.reshape(M, -1).norm(dim=1)[:, None, None]
        self.M = M
        self.N = N

    def forward(self, x):
        out = torch.zeros(x.shape, dtype=x.dtype)
        batch_size = len(x)
        T = (self.psi[None, :, :, :] * self.c[:, :, None, None]).sum(dim=1).reshape((batch_size, self.N, self.N))
        out =torch.matrix_exp(T) @ x
        return out

    def set_coefficients(self, c):
        self.c = c

    def get_psi(self):
        return self.psi.data

    def set_psi(self,psi_input):
        self.psi.data = psi_input


class ZetaDecoder(nn.Module):

    def __init__(self, latent_dim, dict_size):
        super(ZetaDecoder, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1028),
            nn.BatchNorm1d(1028),
            nn.ReLU(),
            nn.Linear(1028, dict_size))

    def forward(self, x):
        return self.model(x)

class ZetaDecoder_small(nn.Module):

    def __init__(self, latent_dim, dict_size):
        super(ZetaDecoder_small, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(latent_dim, dict_size))

    def forward(self, x):
        return self.model(x)
