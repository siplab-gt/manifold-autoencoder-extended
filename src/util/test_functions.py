"""
This script was developed for the purpose of training the manifold autoencoder with various
datasets and methods of point pair supervision. The goal is to effectively learn natural
transformations of datasets (MNIST, CIFAR-10) in the latent space of an autoencoder
using transport operators
"""

import re
import os
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def gen_transopt_paths(encoder, decoder, classifier, transOp,train_loader, t,test_dir,device,stepUse,num_classes,channels,image_dim,args):
    latent_scale = args.latent_scale
    dict_size = args.dict_size
    latent_dim = args.latent_dim
    print("Generating Transport Operator Paths...")
    encoder.eval()
    decoder.eval()
    classifier.eval()

    imgChoice = np.zeros((num_classes,image_dim,image_dim,channels))
    imgDecode = np.zeros((num_classes,image_dim,image_dim,channels))
    x0, _, label = next(iter(train_loader))
    for k in range(0,num_classes):

        idxClass = np.where(label[:] == k)[0]
        idxChoice = idxClass[0]
        imgChoice[k,:,:,:] = x0[idxChoice,:,:,:].permute(1,2,0).detach().numpy()
        imgInput = torch.unsqueeze(x0[idxChoice,:,:,:],0).to(device)


        z = encoder(imgInput)
        z_scale = z/latent_scale

        x_hat = decoder(z)
        imgDecode[k,:,:,:] = x_hat.permute(0,2,3,1).detach().cpu().numpy()

        z_seq = np.zeros((dict_size,len(t),latent_dim))
        img_seq = np.zeros((dict_size,len(t),image_dim,image_dim,channels))
        prob_seq_latent = np.zeros((dict_size,len(t),num_classes))
        prob_seq_img = np.zeros((dict_size,len(t),num_classes))
        for m in range(0,dict_size):
            coeff_use = np.zeros((dict_size))
            t_count = 0
            for t_use in t:
                coeff_use[m] = t_use
                coeff_input = np.expand_dims(coeff_use,axis=0)
                with torch.no_grad():
                    coeff_input = torch.from_numpy(coeff_input).float().to(device)
                transOp.set_coefficients(coeff_input)
                z_trans_scale = torch.unsqueeze(transOp(z_scale.unsqueeze(-1)).squeeze(),0)

                z_trans = z_trans_scale*latent_scale
                #print(z_trans.shape)
                transImgOut = decoder(z_trans)
                prob_out_latent, label_est = classifier(z_trans)
                #prob_out_img,label_est_img = classifier_img(transImgOut)
                #out = simplified_vgg(transImgOut.repeat((1, 3, 1, 1)))
                #prob_out_img = torch.nn.functional.softmax(out, dim=1)[0]
                transImgOut_permute = transImgOut.permute(0,2,3,1)
                transImg_np = transImgOut_permute.detach().cpu().numpy()


                z_seq[m,t_count,:] = z_trans.detach().cpu().numpy()
                img_seq[m,t_count,:,:,:] = transImg_np
                prob_seq_latent[m,t_count,:] = prob_out_latent.detach().cpu().numpy()
                #prob_seq_img[m,t_count,:] = prob_out_img.detach().cpu().numpy()
                t_count = t_count+1
            print("Class " + str(k) + " Operator " + str(m))
        sio.savemat(test_dir + 'transOptOrbitTest_step' + str(stepUse) + '_' + str(k+1) + '.mat',{'latent_seq':z_seq,'imgOut':img_seq,'prob_seq_img':prob_seq_img,
                    'prob_seq_latent':prob_seq_latent,'t_vals':t,'imgChoice':imgChoice,'imgDecode':imgDecode})

def compute_cSpread(encoder,decoder,train_loader,zeta_decoder,test_dir,device,stepUse,batch_size,args):
    latent_dim = args.latent_dim
    dict_size = args.dict_size
    latent_scale = args.latent_scale
    encoder.eval()
    decoder.eval()
    zeta_decoder.eval()
    z_store = np.zeros((5000,latent_dim))
    cspread_store = np.zeros((5000,dict_size))
    label_store = np.zeros((5000))
    for k in range(0,25):
        x0, _, label = next(iter(train_loader))
        x0 = x0.clamp(0, 1).to(device)
        z0 = encoder(x0)/latent_scale
        z_store[k*batch_size:(k+1)*batch_size,:] = z0.detach().cpu().numpy()
        log_spread0 = zeta_decoder(z0)
        coeff_spread0 = 0.1*(0.5*log_spread0).exp()
        cspread_store[k*batch_size:(k+1)*batch_size,:] = coeff_spread0.detach().cpu().numpy()
        label_store[k*batch_size:(k+1)*batch_size] = label.detach().cpu().numpy()

    sio.savemat(test_dir + '/cspreadDecode_step' + str(stepUse) + '.mat',{'z_store':z_store,'cspread_store':cspread_store,'label':label_store})
