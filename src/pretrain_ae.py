"""
This script was developed for the purpose of pre-training an autoencoder that
can be used

@Filename    pretrain_ae.py
@Created     02/03/20
"""

import re
import os
import time
import argparse
import scipy.io as sio

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.autoencoder import init_weights, ConvEncoder, ConvDecoder
from util.dataloader import load_mnist, load_cifar10, load_celeba, load_svhn, load_fmnist

def pretrain_ae(encoder, decoder, train_loader, test_loader, autoenc_opt, autoenc_scheduler, dataset, latent_dim,device,ae_epochs):
    for j in range(ae_epochs):
        for idx, batch in enumerate(train_loader):
            x0, _, __ = batch

            x0 = x0.to(device)
            autoenc_opt.zero_grad()
            z0 = encoder(x0)
            x0_hat = decoder(z0)

            ae_loss = F.mse_loss(x0_hat, x0, reduction='mean')
            ae_loss.backward()
            autoenc_opt.step()

        test_error = torch.zeros(len(test_loader))
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                x0, _, __ = batch
                x0 = x0.to(device)
                z0 = encoder(x0)
                x0_hat = decoder(z0)

                ae_loss = F.mse_loss(x0_hat, x0, reduction='mean')

                test_error[idx] = ae_loss
        print("AE Pre-train epoch {} of {}: Test loss: {:.4E}".format(j+1, ae_epochs, torch.mean(test_error)))
        #if j > 150:
        #    ae_scheduler.step()

    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
    }, './results/pretrained/pretrain_{}_ae_Z{}.pt'.format(dataset, latent_dim))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-Z', '--latent_dim', default=10, type=int, help="Dimension of latent space")
    parser.add_argument('-d', '--dataset', default='mnist', type=str, help="Dataset to Use")
    parser.add_argument('-nae', '--norm_ae_flag', type=int, default=0, help='[0/1] to specify whether to normalize the latent space parameters')
    parser.add_argument('-N', '--train_samples', default=None, type=int, help="Number of training samples to use.")
    parser.add_argument('-c', '--train_classes', default=np.arange(10), nargs='+', type=int, help="Classes to train classifier on.")

    # PARSE ARGUMENTS #
    args = parser.parse_args()
    ae_epochs = 300
    batch_size = 200
    latent_dim = args.latent_dim
    network_lr = 1e-4
    norm_ae_flag = args.norm_ae_flag

    dataset = args.dataset
    train_classes = args.train_classes
    train_samples = args.train_samples
    run_number = args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create folder for logging if it does not exist
    save_dir = args.model_path + dataset + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created directory for figures at {}".format(save_dir))

    # Load training data
    if dataset == 'mnist':
        test_imgs = 10000
        train_loader, test_loader = load_mnist('./data',batch_size, train_samples, test_imgs)
        channels, image_dim, features = 1, 28, 64
        num_classes = len(train_classes)
    elif dataset == 'cifar_vehicle':
        train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=None, train_classes=[0, 1, 8, 9])
        channels, image_dim, features = 1, 32, 256
        num_classes = 4
    elif dataset == 'cifar_animal':
        train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=None, train_classes=[3, 4, 5, 7])
        channels, image_dim, features = 1, 32, 256
        num_classes = 4
    elif dataset == 'cifar10':
        train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
        channels, image_dim, features = 1, 32, 256
        num_classes = len(train_classes)
    elif dataset == 'svhn':
        train_loader, test_loader = load_svhn('./data', batch_size, train_samples =train_samples,train_classes=train_classes)
        channels, image_dim, features = 1, 32, 256
        num_classes = len(train_classes)
    elif dataset == 'fmnist':
        train_loader, test_loader = load_fmnist('./data', batch_size, train_classes=train_classes)
        channels, image_dim, features = 1, 28, 64
        num_classes = len(train_classes)
    elif dataset == 'celeba':
        train_loader, test_loader = load_celeba('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
        class_epochs = 0
        channels, image_dim, features = 1, 32, 256
        num_classes = len(train_classes)

    # Initialize autoencoder model
    encoder.apply(init_weights)
    decoder.apply(init_weights)
    encoder = ConvEncoder(latent_dim, channels, image_dim, norm_ae_flag, dataset=dataset, num_filters=features).to(device)
    decoder = ConvDecoder(latent_dim, channels, image_dim, dataset=dataset, num_filters=features).to(device)
    autoenc_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                   lr=network_lr, betas=(0.5, 0.999))
    autoenc_scheduler = torch.optim.lr_scheduler.ExponentialLR(autoenc_opt, gamma=0.995)

    pretrain_ae(encoder, decoder, train_loader, test_loader, autoenc_opt, autoenc_scheduler, dataset, latent_dim,device,ae_epochs)
