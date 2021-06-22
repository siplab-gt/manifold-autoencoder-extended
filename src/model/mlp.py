#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

class Classifier(nn.Module):

    def __init__(self,y_dim,z_dim):
        """
        Initialize classifier
        Inputs:
        - y_dim : number of classes
        - z_dim: dimension of the latent vector
        """
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512,y_dim)

        
    def forward(self, x):
        """
        Perform classification using the CNN classifier
        
        Inputs:
        - x : input latent vector

        Outputs:
        - output: unnormalized output
        - prob_out: probability output
        """
        x = self.fc1(x)
        fc_out1 = F.relu(x)
        output = self.fc2(fc_out1)
        prob_out = F.softmax(output, dim=1)

        return prob_out,output

