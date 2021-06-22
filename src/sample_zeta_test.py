"""
This script was developed for sampling latent vectors using the encoded
coefficient scale weights and fixed scale weights.

@Filename    path_estimate_test.py
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
from util.dataloader import load_mnist, load_cifar10, load_svhn, load_fmnist, load_celeba
from model.sampler import sample_c

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
parser.add_argument('-css', '--coeff_spread_scale', type=float, default=0.1, help='Scale on coefficient spread weights')
parser.add_argument('-fs', '--fix_spread_scale', type=float, default=0.03, help='Scale on coefficient spread weights')
parser.add_argument('-r', '--run', required=True, type=int, help="Run number")

args = parser.parse_args()

batch_size = 30
latent_scale = args.latent_scale
latent_dim = args.latent_dim
dict_size = args.dict_size
zeta = args.zeta
gamma = args.gamma

coeff_spread_scale = args.coeff_spread_scale
fix_spread_scale = args.fix_spread_scale

dataset = args.dataset
train_classes = args.train_classes
train_samples = args.train_samples
supervision = args.supervision
run_number = args.run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Load training data
if dataset == 'mnist':
    test_imgs = 10000
    train_loader, test_loader = load_mnist('./data/',batch_size, train_samples, test_imgs)
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





if channels == 1:
    from model.classifier import LeNet
    classifier = LeNet(len(train_classes)).to(device)
    classifier_type_temp = 'lenet'
else:
    from model.classifier import CNN
    classifier = CNN(len(train_classes)).to(device)
    classifier_type_temp = 'cnn'



encoder = ConvEncoder(latent_dim, channels, image_dim, 0, num_filters=features).to(device)
decoder = ConvDecoder(latent_dim, channels, image_dim, num_filters=features).to(device)
zeta_decoder = ZetaDecoder(latent_dim, dict_size).to(device)



# Initialize transport operator model
transOp = TransOp_expm(M=dict_size, N=latent_dim, var=0.05).to(device)

# Load model
modelTO_state = torch.load(args.model_path + f'/pretrained/{dataset}/' + 'run{}_modelDict_{}_M{}Z{}_coeffEncode.pt'.format(run_number, dataset, dict_size, latent_dim), map_location=device)
transOp.load_state_dict(modelTO_state['transOp'])
encoder.load_state_dict(modelTO_state['encoder'])
decoder.load_state_dict(modelTO_state['decoder'])
zeta_decoder.load_state_dict(modelTO_state['zeta_decoder'])

# Load classifier
model_state = torch.load('./results/pretrained/pretrain_classifier_{}_{}_{}_ae_Z{}.pt'.format(dataset, classifier_type_temp,'image', latent_dim))
classifier.load_state_dict(model_state['classifier'])

numSamp = 8

encoder.eval()
decoder.eval()
#Encode samples
epoch = iter(test_loader)
x0, _, label = next(epoch)
x0 = x0.to(device)
x0_save = x0.permute(0,2,3,1).detach().cpu().numpy()
z0 = encoder(x0)/latent_scale

# Encode coefficient scale weights
log_spread0 = zeta_decoder(z0)
coeff_spread0 = coeff_spread_scale*(0.5*log_spread0).exp()
coeff_spread_save = coeff_spread0.detach().cpu().numpy()

# Initialize arrays
coeff_spread_fix = fix_spread_scale*torch.ones_like(coeff_spread0)
sampled_x = np.zeros((batch_size,numSamp,image_dim,image_dim, channels))
sampled_x_fix = np.zeros((batch_size,numSamp,image_dim,image_dim, channels))
prob_out_samp = np.zeros((batch_size,numSamp,len(train_classes)))
prob_out_fix = np.zeros((batch_size,numSamp,len(train_classes)))
coeff_samp = np.zeros((batch_size,numSamp,dict_size))
coeff_fix = np.zeros((batch_size,numSamp,dict_size))
for k in range(0,numSamp):
    # Sample with encoded scale weights
    coeff_samp_0 = sample_c(batch_size,dict_size,coeff_spread0)
    coeff_samp[:,k,:] = coeff_samp_0.detach().cpu().numpy()
    transOp.set_coefficients(coeff_samp_0)
    z0_samp = transOp(z0.unsqueeze(-1)).squeeze()
    x0_samp = decoder(z0_samp*latent_scale)
    sampled_x[:,k,:,:,:] = x0_samp.permute(0,2,3,1).detach().cpu().numpy()
    prob_out_img, label_est = classifier(x0_samp)
    prob_out_samp[:,k,:] = prob_out_img.detach().cpu().numpy()

    # Sample with fixed scale weights
    coeff_fix_0 = sample_c(batch_size,dict_size,coeff_spread_fix)
    coeff_fix[:,k,:] = coeff_fix_0.detach().cpu().numpy()
    transOp.set_coefficients(coeff_fix_0)
    z0_rand = transOp(z0.unsqueeze(-1)).squeeze()
    x0_rand = decoder(z0_rand*latent_scale)
    sampled_x_fix[:,k,:,:,:] = x0_rand.permute(0,2,3,1).detach().cpu().numpy()
    prob_out_img_fix, label_est = classifier(x0_rand)
    prob_out_fix[:,k,:] = prob_out_img_fix.detach().cpu().numpy()

sio.savemat(test_dir + '/sampleTest_coeffEncode.mat',{'x0':x0_save,'sampled_x':sampled_x,'params':args,
            'sampled_x_fix':sampled_x_fix,'coeff_spread':coeff_spread_save,'label':label.detach().cpu().numpy(),
            'prob_out_samp':prob_out_samp,'prob_out_fix':prob_out_fix,'coeff_spread_fix':coeff_spread_fix.detach().cpu().numpy(),
            'coeff_samp':coeff_samp,'coeff_fix':coeff_fix})
