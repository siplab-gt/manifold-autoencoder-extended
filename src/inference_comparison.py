#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
import time

import glob, os
import numpy as np
import re
from scipy.io import loadmat
import scipy as sp
import scipy.linalg
import scipy.optimize

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transop import TransOp_expm
from model.l1_inference import infer_coefficients
from model.autoencoder import ConvEncoder, ConvDecoder
from util.dataloader import load_mnist, load_cifar10, load_celeba, load_celeba64

# In[25]:


def transOptObj_c(c,Psi,x0,x1,zeta):
    N = int(np.sqrt(Psi.shape[0]))
    M = int(Psi.shape[1])
    coeff_use = np.expand_dims(c,axis=1)
    x0_use = np.expand_dims(x0,axis=1)
    A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
    T = np.real(sp.linalg.expm(A))
    x1_est= np.dot(T,x0_use)[:,0]
    objFun = 0.5*np.linalg.norm(x1-x1_est)**2 + zeta*np.linalg.norm(c,1)

    return objFun

def transOptDerv_c(c,Psi,x0,x1,zeta):
    N = int(np.sqrt(Psi.shape[0]))
    M = int(Psi.shape[1])
    coeff_use = np.expand_dims(c,axis=1)
    x0_use = np.expand_dims(x0,axis=1)
    x1_use = np.expand_dims(x1,axis=1)
    A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
    T = np.real(sp.linalg.expm(A))

    eig_out = np.linalg.eig(A)
    U = eig_out[1]
    D = eig_out[0]
    V = np.linalg.inv(U)
    V = V.T

    innerVal = np.dot(-x1_use,x0_use.T) + np.dot(T,np.dot(x0_use,x0_use.T))
    P = np.dot(np.dot(U.T,innerVal),V)

    F_mat = np.zeros((D.shape[0],D.shape[0]),dtype=np.complex128)
    for alpha in range(0,D.shape[0]):
        for beta in range(0,D.shape[0]):
            if D[alpha] == D[beta]:
                F_mat[alpha,beta] = np.exp(D[alpha])
            else:
                F_mat[alpha,beta] = (np.exp(D[beta])-np.exp(D[alpha]))/(D[beta]-D[alpha])

    fp = np.multiply(F_mat,P)
    Q1 = np.dot(V,fp)
    Q = np.dot(Q1,U.T)
    c_grad = np.real(np.dot(np.reshape(Q,-1,order='F'),Psi) + + zeta*np.sign(c))
    return c_grad


# In[37]:


def compute_loss(c, x0, x1, psi):
    T = (psi[None, :, :, :] * c[:, :, None, None]).sum(dim=1).reshape((
        x0.shape[0], psi.shape[1], psi.shape[2]))
    x1_hat = torch.matrix_exp(T) @ x0
    loss = F.mse_loss(x1_hat, x1, reduction='sum')
    return loss

def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - zeta) * torch.sign(c)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def infer_prox_coefficients(x0, x1, psi, zeta, c0, max_iter=800, tol=1e-5,
                            device='cpu', acceleration=False):
    c = nn.Parameter(c0, requires_grad=True)
    if acceleration:
        c_opt = torch.optim.SGD([c], lr=1e-2, nesterov=True, momentum=0.9)
    else:
        c_opt = torch.optim.SGD([c], lr=1e-2, nesterov=False)

    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_opt, gamma=0.985)
    change = 1e99
    k = 0
    loss_list = []
    while k < max_iter and change > tol:
        old_coeff = c.clone()

        c_opt.zero_grad()
        loss = compute_loss(c, x0, x1, psi)
        loss.backward()
        c_opt.step()
        opt_scheduler.step()

        with torch.no_grad():
            c.data = soft_threshold(c, get_lr(c_opt)*zeta)

        loss_list.append(loss.item() + zeta*torch.abs(c).sum())
        change = torch.norm(c.data - old_coeff) / (torch.norm(old_coeff) + 1e-9)
        k += 1
    return k, loss_list, c.data

def infer_subg_coefficients(x0, x1, psi, zeta, c0, max_iter=800, tol=1e-5,
                            device='cpu', acceleration=False):
    c = nn.Parameter(c0, requires_grad=True)
    if acceleration:
        c_opt = torch.optim.SGD([c], lr=1e-2, nesterov=True, momentum=0.9)
    else:
        c_opt = torch.optim.SGD([c], lr=1e-2, nesterov=False)

    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_opt, gamma=0.985)
    change = 1e99
    k = 0
    loss_list = []
    while k < max_iter and change > tol:
        old_coeff = c.clone()

        c_opt.zero_grad()
        loss = compute_loss(c, x0, x1, psi)
        loss += zeta*torch.abs(c).sum()
        loss.backward()
        c_opt.step()
        opt_scheduler.step()

        loss_list.append(loss.item())
        change = torch.norm(c.data - old_coeff) / (torch.norm(old_coeff) + 1e-9)
        k += 1
    return k, loss_list, c.data


# In[7]:


latent_dim = "32"
dict_size = "40"
zeta = "0.8"
gamma = "5.2e-07"
aeW = "0.75"
conW = "0.0"
dataset = "celeba64"

run_path = f'results/{dataset}/celeba_M{dict_size}_z{latent_dim}_zeta{zeta}_gamma{gamma}_plr0.001_aeW0.75_conW0.0_resnet_latent_net_rand_VGG_csp0.01_nst50pst50_allC_nonorm2/'
run_name = f'run1_modelDict_cifar10_M{dict_size}Z{latent_dim}zeta{zeta}gam{gamma}_transOpTrain.pt'
run_name = "run1_modelDict_celeba_M40Z32_step2400.pt"
#model_state = torch.load(run_path + run_name)
model_state = torch.load(f'results/{dataset}/{dataset}_M{dict_size}_z{latent_dim}_zeta{zeta}_gamma{gamma}_step2400.pt', map_location='cuda')
img_dim = 64
encoder = ConvEncoder(int(latent_dim), 3, img_dim, False,num_filters=128).to('cuda')
decoder = ConvDecoder(int(latent_dim), 3, img_dim, num_filters=128).to('cuda')
transOp = TransOp_expm(int(dict_size), int(latent_dim)).to('cuda')

encoder.load_state_dict(model_state['encoder'])
decoder.load_state_dict(model_state['decoder'])
transOp.load_state_dict(model_state['transOp'])
dict_mag = model_state['dict_mag']
latent_mag = model_state['latent_mag']
zeta = model_state['zeta']
latent_scale = model_state['latent_scale']
dict_size = int(dict_size)
latent_dim = int(latent_dim)


# In[8]:


train_loader, test_loader = load_celeba64('./data', 100, train_samples=150000)
checkpoint = torch.load(f'results/pretrained/vgg_nn_celeba64_150000_nc10.pt')    
nearest_neighbor = checkpoint['nearest_neighbor'][:, :5]
train_loader.dataset.dataset.set_nn_graph(nearest_neighbor)


# In[42]:


num_trials = 30
tl = iter(train_loader)
loss_dict = {'prox': [], 'acc_prox': [], 'subg': [], 'acc_subg': [], 'gpu_prox': [],
             'gpu_acc_prox': [], 'gpu_subg': [], 'gpu_acc_subg': [], 'cg': []}
time_dict = {'prox': [], 'acc_prox': [], 'subg': [], 'acc_subg': [], 'gpu_prox': [],
             'gpu_acc_prox': [], 'gpu_subg': [], 'gpu_acc_subg': [], 'cg': []}
c_dict = {'prox': [], 'acc_prox': [], 'subg': [], 'acc_subg': [], 'gpu_prox': [],
          'gpu_acc_prox': [], 'gpu_subg': [], 'gpu_acc_subg': []}

psi = transOp.get_psi().detach().cpu()
psig = psi.to('cuda')
psi_use = transOp.get_psi().detach().cpu().reshape(dict_size, -1).T.numpy()


# In[ ]:


for j in range(num_trials):
    
    x0, x1, label = next(tl)
    x0, x1 = x0.to('cuda'), x1.to('cuda')
    z0, z1 = encoder(x0).detach().cpu(), encoder(x1).detach().cpu()
    z0l, z1l = z0.unsqueeze(-1), z1.unsqueeze(-1)
    z0g, z1g = z0l.to('cuda'), z1l.to('cuda')
    c0 = np.random.randn(len(x0), len(psi)) * 0.02
    c0_tensor = torch.tensor(c0).float()
    c0g = c0_tensor.to('cuda')
    
    pre_time = time.time()
    k, loss_list, c = infer_prox_coefficients(z0l, z1l, psi, zeta, c0_tensor)
    infer_time = time.time() - pre_time
    loss_dict['prox'].append(loss_list)
    time_dict['prox'].append(infer_time)
    c_dict['prox'].append(c)
    print(f"Prox Grad Time: {infer_time:.2f} sec")

    pre_time = time.time()
    k, loss_list, c = infer_prox_coefficients(z0l, z1l, psi, zeta, c0_tensor, acceleration=True)
    infer_time = time.time() - pre_time
    loss_dict['acc_prox'].append(loss_list)
    time_dict['acc_prox'].append(infer_time)
    c_dict['acc_prox'].append(c)
    print(f"Acc Prox Grad Time: {infer_time:.2f} sec")

    pre_time = time.time()
    k, loss_list, c = infer_prox_coefficients(z0g, z1g, psig, zeta, c0g, device='cuda')
    infer_time = time.time() - pre_time
    loss_dict['gpu_prox'].append(loss_list)
    time_dict['gpu_prox'].append(infer_time)
    c_dict['gpu_prox'].append(c)
    print(f"GPU Prox Grad Time: {infer_time:.2f} sec")

    pre_time = time.time()
    k, loss_list, c = infer_prox_coefficients(z0g, z1g, psig, zeta, c0g, acceleration=True, device='cuda')
    infer_time = time.time() - pre_time
    loss_dict['gpu_acc_prox'].append(loss_list)
    time_dict['gpu_acc_prox'].append(infer_time)
    c_dict['gpu_acc_prox'].append(c)
    print(f"GPU Acc Prox Grad Time: {infer_time:.2f} sec")

    pre_time = time.time()
    k, loss_list, c = infer_subg_coefficients(z0l, z1l, psi, zeta, c0_tensor)
    infer_time = time.time() - pre_time
    loss_dict['subg'].append(loss_list)
    time_dict['subg'].append(infer_time)
    c_dict['subg'].append(c)
    print(f"Subg Grad Time: {infer_time:.2f} sec")

    pre_time = time.time()
    k, loss_list, c = infer_subg_coefficients(z0l, z1l, psi, zeta, c0_tensor, acceleration=True)
    infer_time = time.time() - pre_time
    loss_dict['acc_subg'].append(loss_list)
    time_dict['acc_subg'].append(infer_time)
    c_dict['acc_subg'].append(c)
    print(f"Acc Subg Grad Time: {infer_time:.2f} sec")

    pre_time = time.time()
    k, loss_list, c = infer_subg_coefficients(z0g, z1g, psig, zeta, c0g, device='cuda')
    infer_time = time.time() - pre_time
    loss_dict['gpu_subg'].append(loss_list)
    time_dict['gpu_subg'].append(infer_time)
    c_dict['gpu_subg'].append(c)
    print(f"GPU Subg Grad Time: {infer_time:.2f} sec")

    pre_time = time.time()
    k, loss_list, c = infer_subg_coefficients(z0g, z1g, psig, zeta, c0g, acceleration=True, device='cuda')
    infer_time = time.time() - pre_time
    loss_dict['gpu_acc_subg'].append(loss_list)
    time_dict['gpu_acc_subg'].append(infer_time)
    c_dict['gpu_acc_subg'].append(c)
    print(f"GPU Acc Subg Grad Time: {infer_time:.2f} sec")
    
    z0_cg, z1_cg = z0.detach().cpu().numpy(), z1.detach().cpu().numpy()
    pre_time = time.time()
    for j in range(len(x0)):
        opt_out = sp.optimize.minimize(transOptObj_c,c0[j],args=(psi_use,z0_cg[j], z1_cg[j], zeta),
                                       method = 'CG', jac=transOptDerv_c,options={'maxiter':200,'disp':False},
                                       tol = 10^-10)
    infer_time = time.time() - pre_time
    time_dict['cg'].append(infer_time)
    print(f"Total CG Time: {infer_time:.2f}sec")
    print()

    torch.save({'loss_dict': loss_dict,
                'time_dict': time_dict,
                'c_dict' c_dict}, 'results/inference_comp.pt') 



