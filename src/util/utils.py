#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:24:51 2019
"""

from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import glob
import scipy as sp
import os
import scipy.io as sio
import codecs
import gzip

import torch
from sklearn.neighbors import NearestNeighbors

def print_statistics(dict_size, coefficients, c_loss, k, psi):
    count_nz = np.zeros(dict_size+1, dtype=int)
    coeff_np = coefficients.detach().cpu().numpy()
    coeff_nz = np.count_nonzero(coeff_np, axis=0)
    nz_tot = np.count_nonzero(coeff_nz)
    total_nz = np.count_nonzero(coeff_np, axis=1)
    psi_norm = psi.reshape(dict_size, -1).norm(dim=1)
    for z in range(len(total_nz)):
        count_nz[total_nz[z]] += 1
    large_coeff = len(coefficients) - count_nz[:10].sum()
    print("Non-zero elements per bin: {}".format(count_nz))
    #print("Elements using >10 coeff: {}".format(large_coeff))
    #print("Non-zero by coefficient: {}".format(coeff_nz))
    print("Non-zero by coefficient #: {}".format(nz_tot))
    print("Final coefficient loss: {:.3E}".format(c_loss))
    #print("Avg c grad: {}".format(c_grad))
    print("Avg c norms: {:.3E}".format(coefficients.mean(dim=0).norm()))
    print("Avg tensor norms: {:.3E}".format(psi_norm.mean()))
    #print("Number of iterations: {}\n".format(k))
    return (count_nz, nz_tot, psi_norm)

# Build nearest neighbor graph in latent space of encoder
def build_vgg_nn_graph(train_loader, latent_dim, resnet, neighbor_count=5, device='cuda'):
    latent_points = np.zeros((len(train_loader.dataset), 512))
    no_shuffle_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                    batch_size=100,
                                                    shuffle=False,
                                                    num_workers=2)
    for i, batch in enumerate(no_shuffle_loader):
        x0, x1, label = batch
        x0 = x0.to(device)
        if x0.shape[1] == 1:
            x0 = x0.repeat((1, 3, 1, 1))
        latent_points[i*100:(i+1)*100] = resnet(x0).squeeze().detach().cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=neighbor_count+1, algorithm='ball_tree').fit(latent_points)
    return nbrs.kneighbors(latent_points, neighbor_count+1, return_distance=False)[:, 1:]

# Build nearest neighbor graph in latent space of encoder
def build_labeled_vgg_nn_graph(train_loader, latent_dim, resnet, num_classes, neighbor_count=5, device='cuda'):
    latent_points = np.zeros((len(train_loader.dataset), 512))
    labels = np.zeros(len(train_loader.dataset))
    nn = np.zeros((len(train_loader.dataset), neighbor_count))

    no_shuffle_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                    batch_size=100,
                                                    shuffle=False,
                                                    num_workers=2)
    for i, batch in enumerate(no_shuffle_loader):
        x0, x1, label = batch
        x0 = x0.to(device)
        if x0.shape[1] == 1:
            x0 = x0.repeat((1, 3, 1, 1))
        latent_points[i*100:(i+1)*100] = resnet(x0).squeeze().detach().cpu().numpy()
        labels[i*100:(i+1)*100] = label.detach().cpu().numpy()

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        class_points = latent_points[class_idx]
        nbrs = NearestNeighbors(n_neighbors=neighbor_count+1, algorithm='ball_tree').fit(class_points)
        nn[class_idx] = class_idx[nbrs.kneighbors(class_points, neighbor_count+1, return_distance=False)[:, 1:]]

    return nn

def build_label_graph(train_loader, latent_dim, simplified_vgg, num_classes, neighbor_count, device):
    nn = np.zeros((len(train_loader.dataset), neighbor_count))
    labels = np.array(train_loader.dataset.targets)

    for i in range(len(nn)):
        candidates = np.where(labels == labels[i])
        nn[i] = np.random.choice(candidates, size=neighbor_count)

    return nn

# Build nearest neighbor graph in latent space of encoder
def build_nn_graph(train_loader, latent_dim, encoder, neighbor_count=5, device='cuda'):
    latent_points = np.zeros((len(train_loader.dataset), latent_dim))
    no_shuffle_loader = torch.utils.data.DataLoader(train_loader.dataset,
                                                    batch_size=100,
                                                    shuffle=False,
                                                    num_workers=2)
    for i, batch in enumerate(no_shuffle_loader):
        x0, x1, _ = batch
        x0 = x0.to(device)
        latent_points[i*100:(i+1)*100] = encoder(x0).detach().cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=neighbor_count+1, algorithm='ball_tree').fit(latent_points)
    return nbrs.kneighbors(latent_points, neighbor_count+1, return_distance=False)[:, 1:]
