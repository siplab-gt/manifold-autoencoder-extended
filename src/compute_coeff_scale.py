"""
This script was developed for the purpose of computing the coefficient scale
values for latent vectors in the dataset

@Filename    train_transop_MNIST.py
@Created     02/03/20
"""

import os
import argparse

import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.transop import TransOp_expm, ZetaDecoder
from model.l1_inference import infer_coefficients
from model.autoencoder import init_weights, ConvEncoder, ConvDecoder
from model.classifier import SimplifiedResNet
from util.utils import build_vgg_nn_graph, build_nn_graph, print_statistics
from util.dataloader import load_mnist, load_cifar10, load_svhn, load_fmnist, load_index_dataset, load_celeba

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./results/', help='folder name')
parser.add_argument('-Z', '--latent_dim', default=32, type=int, help="Dimension of latent space")
parser.add_argument('-M', '--dict_size', default=30, type=int, help="Dictionary size")
parser.add_argument('-d', '--dataset', default='cifar10', type=str, help="Dataset to Use")
parser.add_argument('-s', '--supervision', default='NN', type=str, help="Default supervision method for selecting point pairs")
parser.add_argument('-z', '--zeta', default=0.5, type=float, help="Zeta L1 reg")
parser.add_argument('-g', '--gamma', default=2e-6, type=float, help="Gamma L2 reg")
parser.add_argument('-l', '--latent_scale', default=30.0, type=float, help="Scaling term in latent space")
parser.add_argument('-N', '--train_samples', default=50000, type=int, help="Number of training samples to use.")
parser.add_argument('-c', '--train_classes', default=np.arange(10), nargs='+', type=int, help="Classes to train classifier on.")
parser.add_argument('-r', '--run', required=True, type=int, help="Run number")
parser.add_argument('-p', '--pretrain', action='store_true', help="Use pretrained autoencoder")
parser.add_argument('-st', '--save_test', action='store_true', help="Save test data")

args = parser.parse_args()

# Initialize parameters
batch_size = 100
latent_scale = args.latent_scale
latent_dim = args.latent_dim
dict_size = args.dict_size
zeta = args.zeta
gamma = args.gamma
dataset = args.dataset
train_classes = args.train_classes
train_samples = args.train_samples
supervision = args.supervision

run_number = args.run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create directories
save_dir = args.model_path + '/' + dataset + '/' + dataset  + '_zetaNet_M' + str(dict_size) + '_z' + str(latent_dim)
test_dir = save_dir + '_test/'
save_dir = save_dir + '/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

if args.save_test:
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print("Created directory for test data at {}".format(test_dir))

# Load data
if dataset == 'mnist':
    test_imgs = 10000
    #train_loader, test_loader = load_mnist('./data/',batch_size, train_samples, test_imgs)
    test_loader = load_index_dataset('./data/', batch_size, np.arange(10000), dataset="mnist", data_type = 'test')
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
elif dataset == 'cifar10_vehicle':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[0, 1, 8, 9])
    channels, image_dim, features = 3, 32, 256
    num_classes = 4
elif dataset == 'cifar10_animal':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[3, 4, 5, 7])
    class_epochs = 0
    channels, image_dim, features = 3, 32, 256
    num_classes = 4
elif dataset == 'cifar10':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    channels, image_dim, features = 3, 32, 256
    num_classes = len(train_classes)
elif dataset == 'svhn':
    #train_loader, test_loader = load_svhn('./data', batch_size, train_samples =train_samples,train_classes=train_classes)
    test_loader = load_index_dataset('./data/', batch_size, np.arange(10000), dataset="svhn", data_type = 'test')
    channels, image_dim, features = 3, 32, 256
    num_classes = len(train_classes)
elif dataset == 'fmnist':
    #train_loader, test_loader = load_fmnist('./data', batch_size, train_classes=train_classes)
    test_loader = load_index_dataset('./data/', batch_size, np.arange(10000), dataset="fmnist", data_type = 'test')
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
elif dataset == 'celeba':
    train_loader, test_loader = load_celeba('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    class_epochs = 0
    channels, image_dim, features = 3, 32, 256
    num_classes = len(train_classes)


# Initialize networks
encoder = ConvEncoder(latent_dim, channels, image_dim, 0, num_filters=features).to(device)
decoder = ConvDecoder(latent_dim, channels, image_dim, num_filters=features).to(device)
zeta_decoder = ZetaDecoder(latent_dim, dict_size).to(device)
transOp = TransOp_expm(M=dict_size, N=latent_dim, var=0.05).to(device)

# Load model
modelTO_state = torch.load(args.model_path + f'/pretrained/{dataset}/' + 'run{}_modelDict_{}_M{}Z{}_coeffEncode.pt'.format(run_number, dataset, dict_size, latent_dim), map_location=device)
transOp.load_state_dict(modelTO_state['transOp'])
encoder.load_state_dict(modelTO_state['encoder'])
decoder.load_state_dict(modelTO_state['decoder'])
zeta_decoder.load_state_dict(modelTO_state['zeta_decoder'])


encoder.eval()
decoder.eval()
numStep = 50

z_store = np.zeros((numStep*batch_size,latent_dim))
cspread_store = np.zeros((numStep*batch_size,dict_size))
imgUse = np.zeros((1000,image_dim,image_dim,channels))
label_store = np.zeros((numStep*batch_size))
index_store = np.zeros((numStep*batch_size))
epoch = iter(test_loader)
for k in range(0,numStep):
    # Encode data
    x0, label,index = next(epoch)
    x0 = x0.to(device)
    z0 = encoder(x0)/latent_scale
    z_store[k*batch_size:(k+1)*batch_size,:] = z0.detach().cpu().numpy()
    #Encode scale weights
    log_spread0 = zeta_decoder(z0)
    coeff_spread0 = 0.1*(0.5*log_spread0).exp()
    cspread_store[k*batch_size:(k+1)*batch_size,:] = coeff_spread0.detach().cpu().numpy()
    label_store[k*batch_size:(k+1)*batch_size] = label.detach().cpu().numpy()
    index_store[k*batch_size:(k+1)*batch_size] = index.detach().cpu().numpy()
    if k < 10:
        imgUse[k*batch_size:(k+1)*batch_size,:,:,:] = x0.permute(0,2,3,1).detach().cpu().numpy()

sio.savemat(test_dir + '/coeffScale_coeffEncode.mat',{'z_store':z_store,'cspread_store':cspread_store,'label':label_store,'index':index_store,'imgUse':imgUse})
