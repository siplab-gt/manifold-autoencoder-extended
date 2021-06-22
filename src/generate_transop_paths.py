"""
This script was developed for the purpose of generating transformation paths
with transport operators learned in the manifold autoencoder.

@Filename    generate_transop_paths_allData.py
@Created     02/03/20
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

from model.transop import TransOp_expm
from model.l1_inference import infer_coefficients
from model.autoencoder import init_weights, ConvEncoder, ConvDecoder, ConvEncoder_old, ConvDecoder_old
from model.classifier import SimplifiedResNet
from util.utils import build_vgg_nn_graph, build_nn_graph, print_statistics
from util.dataloader import load_mnist, load_cifar10, load_svhn, load_fmnist,load_celeba, load_celeba64

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./results', help='folder name')
parser.add_argument('-Z', '--latent_dim', default=32, type=int, help="Dimension of latent space")
parser.add_argument('-M', '--dict_size', default=30, type=int, help="Dictionary size")
parser.add_argument('-d', '--dataset', default='cifar10', type=str, help="Dataset to Use")
parser.add_argument('-s', '--supervision', default='NN', type=str, help="Default supervision method for selecting point pairs")
parser.add_argument('-z', '--zeta', default=0.5, type=float, help="Zeta L1 reg")
parser.add_argument('-g', '--gamma', default=2e-6, type=float, help="Gamma L2 reg")
parser.add_argument('-l', '--latent_scale', default=30.0, type=float, help="Scaling term in latent space")
parser.add_argument('-N', '--train_samples', default=50000, type=int, help="Number of training samples to use.")
parser.add_argument('-nr', '--num_restart', default=1, type=int, help="Number of restarts for coefficient inference.")
parser.add_argument('-c', '--train_classes', default=np.arange(10), nargs='+', type=int, help="Classes to train classifier on.")
parser.add_argument('-r', '--run', required=True, type=int, help="Run number")
parser.add_argument('-nonneg', action='store_true', help="Non-negative constraint on coefficients.")
parser.add_argument('-st', '--save_test', action='store_true', help="Save test data")

args = parser.parse_args()

# Initialize parameters
batch_size = 100
latent_scale = args.latent_scale
latent_dim = args.latent_dim
dict_size = args.dict_size
zeta = args.zeta
gamma = args.gamma
num_restart = args.num_restart
dataset = args.dataset
train_classes = args.train_classes
train_samples = args.train_samples
supervision = args.supervision
run_number = args.run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


save_dir = args.model_path + '/' + dataset + '/' + dataset  + '_M' + str(dict_size) + '_z' + str(latent_dim) + '_zeta' + str(zeta) + '_gamma' + str(gamma)

test_dir = save_dir + '_test/'
save_dir = save_dir + '/'



if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

if args.save_test:
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print("Created directory for test data at {}".format(test_dir))



# Load training data
if dataset == 'mnist':
    test_imgs = 10000
    train_loader, test_loader = load_mnist('./data',batch_size, train_samples, test_imgs)
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
elif dataset == 'cifar10_vehicle':
    train_samples = None
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[0, 1, 8, 9])
    channels, image_dim, features = 3, 32, 256
    num_classes = 4
elif dataset == 'cifar10_animal':
    train_samples = None
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[3, 4, 5, 7])
    class_epochs = 0
    channels, image_dim, features = 3, 32, 256
    num_classes = 4
elif dataset == 'cifar10':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    channels, image_dim, features = 3, 32, 256
    num_classes = len(train_classes)
elif dataset == 'svhn':
    train_loader, test_loader = load_svhn('./data', batch_size, train_samples =train_samples,train_classes=train_classes)
    channels, image_dim, features = 3, 32, 256
    num_classes = len(train_classes)
elif dataset == 'fmnist':
    train_loader, test_loader = load_fmnist('./data', batch_size, train_classes=train_classes)
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
elif dataset == 'celeba':
    train_loader, test_loader = load_celeba('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    class_epochs = 0
    channels, image_dim, features = 3, 32, 256
    num_classes = len(train_classes)
elif dataset == 'celeba64':
    batch_size = 500
    train_loader, test_loader = load_celeba64('/storage/coda1/p-crozell3/0/shared/Manifold_Autoencoder_Transfer/data', batch_size, train_samples=train_samples, train_classes=train_classes)
    class_epochs = 0
    channels, image_dim, features = 3, 64, 128
    num_classes = len(train_classes)

# Load autoencoder
if dataset == 'svhn':
    encoder = ConvEncoder_old(latent_dim, channels, image_dim, 0, num_filters=features).to(device)
    decoder = ConvDecoder_old(latent_dim, channels, image_dim, num_filters=features).to(device)
else:
    encoder = ConvEncoder(latent_dim, channels, image_dim, 0, num_filters=features).to(device)
    decoder = ConvDecoder(latent_dim, channels, image_dim, num_filters=features).to(device)


# Initialize transport operator model
transOp = TransOp_expm(M=dict_size, N=latent_dim, var=0.05).to(device)

# Load model
t = np.arange(-0.625*1.0,0.65625*1.0,0.03125*1.0)
modelTO_state = torch.load(args.model_path + '/pretrained/' + dataset + '/run{}_modelDict_{}_M{}Z{}_finetune.pt'.format(run_number, dataset, dict_size, latent_dim), map_location=device)
transOp.load_state_dict(modelTO_state['transOp'])
encoder.load_state_dict(modelTO_state['encoder'])
decoder.load_state_dict(modelTO_state['decoder'])

encoder.eval()
decoder.eval()

# Normalize the transport operators
normalize_val = 3.0
psi_new= transOp.get_psi()
psi_norm = torch.diag(torch.div(normalize_val,torch.sqrt(torch.sum(torch.square(torch.reshape(psi_new,(dict_size,latent_dim*latent_dim))),axis = 1))+1e-11))
psi_new_norm = torch.matmul(torch.transpose(torch.reshape(psi_new,(dict_size,latent_dim*latent_dim)),0,1),psi_norm)
psi_square = torch.reshape(torch.transpose(psi_new_norm,0,1),(dict_size,latent_dim,latent_dim))
transOp.set_psi(psi_square)

imgChoice = np.zeros((num_classes,image_dim,image_dim,channels))
x0, _, label = next(iter(test_loader))
for k in range(0,num_classes):

    # Load sample from selected class
    idxClass = np.where(label[:] == k)[0]
    idxChoice = idxClass[0]
    imgChoice[k,:,:,:] = x0[idxChoice,:,:,:].permute(1,2,0).detach().numpy()
    imgInput = torch.unsqueeze(x0[idxChoice,:,:,:],0).to(device)

    #Encode and scale data
    z = encoder(imgInput)
    z_scale = z/latent_scale

    z_seq = np.zeros((dict_size,len(t),latent_dim))
    img_seq = np.zeros((dict_size,len(t),image_dim,image_dim,channels))
    for m in range(0,dict_size):
        coeff_use = np.zeros((dict_size))
        t_count = 0
        for t_use in t:
            # Vary the value of a single coefficient value
            coeff_use[m] = t_use
            coeff_input = np.expand_dims(coeff_use,axis=0)
            with torch.no_grad():
                coeff_input = torch.from_numpy(coeff_input).float().to(device)

            # Apply the transport operators to the latent vector
            transOp.set_coefficients(coeff_input)
            z_trans_scale = torch.unsqueeze(transOp(z_scale.unsqueeze(-1)).squeeze(),0)

            z_trans = z_trans_scale*latent_scale
            transImgOut = decoder(z_trans)

            # Store data outputs
            transImgOut_permute = transImgOut.permute(0,2,3,1)
            transImg_np = transImgOut_permute.detach().cpu().numpy()
            z_seq[m,t_count,:] = z_trans.detach().cpu().numpy()
            img_seq[m,t_count,:,:,:] = transImg_np
            t_count = t_count+1
        print("Class " + str(k) + " Operator " + str(m))
    sio.savemat(test_dir + 'transOptOrbitTest_finetune_' + str(k+1) + '.mat',{'latent_seq':z_seq,'imgOut':img_seq,'t_vals':t,'imgChoice':imgChoice})
