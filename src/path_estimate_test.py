"""
This script was developed for the computing interpolated and extrapolated paths
using transport operators.

@Filename    path_estimate_test.py
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

from model.transop import TransOp_expm, ZetaDecoder
from model.l1_inference import infer_coefficients, compute_arc_length
from model.autoencoder import init_weights, ConvEncoder, ConvDecoder,BetaVAE
from model.classifier import SimplifiedResNet
from util.utils import build_vgg_nn_graph, build_nn_graph, print_statistics
from util.dataloader import load_mnist, load_cifar10, load_svhn, load_fmnist

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./results/', help='folder name')
parser.add_argument('-Z', '--latent_dim', default=10, type=int, help="Dimension of latent space")
parser.add_argument('-M', '--dict_size', default=16, type=int, help="Dictionary size")
parser.add_argument('-d', '--dataset', default='mnist', type=str, help="Dataset to Use")
parser.add_argument('-s', '--supervision', default='VGG', type=str, help="Default supervision method for selecting point pairs")
parser.add_argument('-z', '--zeta', default=0.05, type=float, help="Zeta L1 reg")
parser.add_argument('-g', '--gamma', default=2e-5, type=float, help="Gamma L2 reg")
parser.add_argument('-l', '--latent_scale', default=30.0, type=float, help="Scaling term in latent space")
parser.add_argument('-m', '--psi_var', default=.05, type=float, help="Variance to scale psi with")
parser.add_argument('-N', '--train_samples', default=2000, type=int, help="Number of training samples to use.")
parser.add_argument('-nr', '--num_restart', default=1, type=int, help="Number of restarts for coefficient inference.")
parser.add_argument('-c', '--train_classes', default=np.arange(10), nargs='+', type=int, help="Classes to train classifier on.")
parser.add_argument('-transfer_mod', type=int, default=4, help='Number of steps on only net updates during transfer')
parser.add_argument('-imgFlag', type=int, default=0, help='Flag to determine if you want to limit the batch size and save images or use large batch sizes and save off only accuracy')
parser.add_argument('-step', '--stepUse', default=25000, type=int, help="Training step to test on.")
parser.add_argument('-r', '--run', required=True, type=int, help="Run number")
parser.add_argument('-st', '--save_test', action='store_true', help="Save test data")

args = parser.parse_args()
imgFlag = args.imgFlag
if imgFlag == 1:
    batch_size = 10
else:
    batch_size = 400
latent_scale = args.latent_scale
latent_dim = args.latent_dim
dict_size = args.dict_size
zeta = args.zeta
gamma = args.gamma
psi_var = args.psi_var
num_restart = args.num_restart
dataset = args.dataset
train_classes = args.train_classes
train_samples = args.train_samples
supervision = args.supervision


run_number = args.run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_int = 0.01
t_vals = np.arange(0.0,1.0+t_int,t_int)
# Define path length for interpolation/extrapolation
t_path_long = np.arange(0.0,3.3,0.3)

# Specify directories with saved files
save_dir = args.model_path + '/' + dataset + '/' + dataset  + '_M' + str(dict_size) + '_z' + str(latent_dim) + '_zeta' + str(zeta) + '_gamma' + str(gamma)
test_dir = save_dir + '_test/'
save_dir = save_dir + '/'
save_dir_CAE = f'./results/CAE_{dataset}_{latent_dim}/'
save_dir_BVAE = f'./results/BVAE_{dataset}_{latent_dim}/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

if args.save_test:
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print("Created directory for test data at {}".format(test_dir))

# Load training data (one loader per class)
if dataset == 'mnist':
    test_imgs = 10
    train_loader0, test_loader0 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [0] )
    train_loader1, test_loader1 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [1] )
    train_loader2, test_loader2 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [2] )
    train_loader3, test_loader3 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [3] )
    train_loader4, test_loader4 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [4] )
    train_loader5, test_loader5 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [5] )
    train_loader6, test_loader6 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [6] )
    train_loader7, test_loader7 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [7] )
    train_loader8, test_loader8 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [8] )
    train_loader9, test_loader9 = load_mnist('./data/',batch_size, train_samples, test_imgs,train_classes = [9] )
    image_dim = 28
    num_classes = len(train_classes)
    features = 64
    channels = 1
    encoder = ConvEncoder(latent_dim, channels, image_dim,0).to(device)
    encoder_ae = ConvEncoder(latent_dim, channels, image_dim,0).to(device)
    decoder = ConvDecoder(latent_dim, channels, image_dim).to(device)
    decoder_ae = ConvDecoder(latent_dim, channels, image_dim).to(device)
elif dataset == 'cifar10':
    train_loader0, test_loader0 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [0] )
    train_loader1, test_loader1 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [1] )
    train_loader2, test_loader2 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [2] )
    train_loader3, test_loader3 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [3] )
    train_loader4, test_loader4 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [4] )
    train_loader5, test_loader5 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [5] )
    train_loader6, test_loader6 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [6] )
    train_loader7, test_loader7 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [7] )
    train_loader8, test_loader8 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [8] )
    train_loader9, test_loader9 = load_cifar10('./data',batch_size, train_samples =train_samples, train_classes = [9] )
    channels = 3
    image_dim = 32
    features = 256
    num_classes = len(train_classes)
    encoder = ConvEncoder(latent_dim, channels, image_dim,0, num_filters=256).to(device)
    encoder_ae = ConvEncoder(latent_dim, channels, image_dim,0, num_filters=256).to(device)
    decoder = ConvDecoder(latent_dim, channels, image_dim,  num_filters=256).to(device)
    decoder_ae = ConvDecoder(latent_dim, channels, image_dim,  num_filters=256).to(device)
elif dataset == 'svhn':
    train_loader0, test_loader0 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [0] )
    train_loader1, test_loader1 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [1] )
    train_loader2, test_loader2 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [2] )
    train_loader3, test_loader3 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [3] )
    train_loader4, test_loader4 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [4] )
    train_loader5, test_loader5 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [5] )
    train_loader6, test_loader6 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [6] )
    train_loader7, test_loader7 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [7] )
    train_loader8, test_loader8 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [8] )
    train_loader9, test_loader9 = load_svhn('./data',batch_size, train_samples =train_samples, train_classes = [9] )

    channels = 3
    image_dim = 32
    features = 256
    num_classes = len(train_classes)
    encoder = ConvEncoder(latent_dim, channels, image_dim,0, num_filters=256).to(device)
    encoder_ae = ConvEncoder(latent_dim, channels, image_dim,0, num_filters=256).to(device)
    decoder = ConvDecoder(latent_dim, channels, image_dim, num_filters=256).to(device)
    decoder_ae = ConvDecoder(latent_dim, channels, image_dim, num_filters=256).to(device)
elif dataset == 'fmnist':
    train_loader0, test_loader0 = load_fmnist('./data',batch_size, train_classes = [0] )
    train_loader1, test_loader1 = load_fmnist('./data',batch_size, train_classes = [1] )
    train_loader2, test_loader2 = load_fmnist('./data',batch_size, train_classes = [2] )
    train_loader3, test_loader3 = load_fmnist('./data',batch_size, train_classes = [3] )
    train_loader4, test_loader4 = load_fmnist('./data',batch_size, train_classes = [4] )
    train_loader5, test_loader5 = load_fmnist('./data',batch_size, train_classes = [5] )
    train_loader6, test_loader6 = load_fmnist('./data',batch_size, train_classes = [6] )
    train_loader7, test_loader7 = load_fmnist('./data',batch_size, train_classes = [7] )
    train_loader8, test_loader8 = load_fmnist('./data',batch_size, train_classes = [8] )
    train_loader9, test_loader9 = load_fmnist('./data',batch_size, train_classes = [9] )
    image_dim = 28
    num_classes = len(train_classes)
    channels = 1
    features = 64
    encoder = ConvEncoder(latent_dim, channels, image_dim,0).to(device)
    encoder_ae = ConvEncoder(latent_dim, channels, image_dim,0).to(device)
    decoder = ConvDecoder(latent_dim, channels, image_dim).to(device)
    decoder_ae = ConvDecoder(latent_dim, channels, image_dim).to(device)


# Load AE model
modelAE_state = torch.load('./results/pretrained/pretrain_{}_ae_Z{}.pt'.format(dataset, latent_dim),  map_location=device)
encoder_ae.load_state_dict(modelAE_state['encoder'])
decoder_ae.load_state_dict(modelAE_state['decoder'])

# Load CAE model
encoder_cae = ConvEncoder(latent_dim, channels, image_dim, False, num_filters=features).to(device)
decoder_cae = ConvDecoder(latent_dim, channels, image_dim, num_filters=features).to(device)
Lambda = 1e-4
modelCAE_state = torch.load(save_dir_CAE + 'CAE_{}_Z{}_Lambda{}.pt'.format(dataset, latent_dim, Lambda),  map_location=device)
encoder_cae.load_state_dict(modelCAE_state['encoder'])
decoder_cae.load_state_dict(modelCAE_state['decoder'])

# Load beta-VAE model
bvae = BetaVAE(z_dim=latent_dim, nc=channels, img_size = image_dim).to(device)
Beta = 5
modelBVAE_state = torch.load(save_dir_BVAE + 'BVAE_{}_Z{}_Beta{}.pt'.format(dataset, latent_dim, Beta),  map_location=device)
bvae.load_state_dict(modelBVAE_state['bvae'])

zeta_decoder = ZetaDecoder(latent_dim, dict_size).to(device)

# Initialize image classifier
if channels == 1:
    from model.classifier import LeNet
    classifier = LeNet(len(train_classes)).to(device)
    classifier_type_temp = 'lenet'
else:
    from model.classifier import CNN
    classifier = CNN(len(train_classes)).to(device)
    classifier_type_temp = 'cnn'
model_state = torch.load('./results/pretrained/pretrain_classifier_{}_{}_{}_ae_Z{}.pt'.format(dataset, classifier_type_temp,'image', latent_dim), map_location=device)
classifier.load_state_dict(model_state['classifier'])

# Initialize transport operator model
transOp = TransOp_expm(M=dict_size, N=latent_dim, var=psi_var).to(device)

# Load MAE model
stepUse = args.stepUse
t = np.arange(-0.625*4,0.65625*4,0.03125*4/2)
modelTO_state = torch.load(args.model_path + f'/pretrained/{dataset}/' + 'run{}_modelDict_{}_M{}Z{}_coeffEncode.pt'.format(run_number, dataset, dict_size, latent_dim), map_location=device)
transOp.load_state_dict(modelTO_state['transOp'])
encoder.load_state_dict(modelTO_state['encoder'])
decoder.load_state_dict(modelTO_state['decoder'])
zeta_decoder.load_state_dict(modelTO_state['zeta_decoder'])


encoder.eval()
decoder.eval()
encoder_ae.eval()
decoder_ae.eval()
encoder_cae.eval()
decoder_cae.eval()
bvae.eval()
zeta_decoder.eval()
classifier.eval()

# Run tests when the classes are the same
num_restart_test = 10
num_samp = 10
for k in range(0,num_classes):
    if k == 0:
        test_loader = test_loader0
    elif k == 1:
        test_loader = test_loader1
    elif k == 2:
        test_loader = test_loader2
    elif k == 3:
        test_loader = test_loader3
    elif k == 4:
        test_loader = test_loader4
    elif k == 5:
        test_loader = test_loader5
    elif k == 6:
        test_loader = test_loader6
    elif k ==7:
        test_loader = test_loader7
    elif k == 8:
        test_loader = test_loader8
    elif k == 9:
        test_loader = test_loader9

    # Initialize arrays
    if imgFlag == 1:
        transImgTotal = np.zeros((len(t_path_long),batch_size,image_dim,image_dim,channels))
        z_trans_out = np.zeros((len(t_path_long),batch_size,latent_dim))
        transImgTotal_euc = np.zeros((len(t_path_long),batch_size,image_dim,image_dim,channels))
        z_trans_out_euc = np.zeros((len(t_path_long),batch_size,latent_dim))
        transImgTotal_cae = np.zeros((len(t_path_long),batch_size,image_dim,image_dim,channels))
        z_trans_out_cae = np.zeros((len(t_path_long),batch_size,latent_dim))
        transImgTotal_bvae = np.zeros((len(t_path_long),batch_size,image_dim,image_dim,channels))
        z_trans_out_bvae = np.zeros((len(t_path_long),batch_size,latent_dim))
        c_spread_path = np.zeros((len(t_path_long),batch_size,dict_size))
        x1_out = np.zeros((batch_size,image_dim,image_dim,channels))
    acc_out_euc = np.zeros((len(t_path_long),batch_size))
    acc_out_cae = np.zeros((len(t_path_long),batch_size))
    acc_out_bvae = np.zeros((len(t_path_long),batch_size))
    acc_out = np.zeros((len(t_path_long),batch_size))
    prob_out_euc = np.zeros((len(t_path_long),batch_size))
    prob_out_cae = np.zeros((len(t_path_long),batch_size))
    prob_out_bvae = np.zeros((len(t_path_long),batch_size))
    prob_out = np.zeros((len(t_path_long),batch_size))


    # Load input data
    x0,_,label = next(iter(test_loader))
    x1,_,label_comp = next(iter(test_loader))
    x0= x0.to(device)
    x1 = x1.to(device)
    label = label.to(device)
    label_comp = label_comp.to(device)
    label_true = torch.ones_like(label).to(device)*k

    # Encode data
    z0 = encoder(x0)/latent_scale
    z0_ae = encoder_ae(x0)/latent_scale
    z0_cae = encoder_cae(x0)/latent_scale
    _,z0_bvae,_ = bvae(x0)
    z0_bvae = z0_bvae/latent_scale
    z0_store = z0.detach().cpu().numpy()
    z1 = encoder(x1)/latent_scale
    z1_ae = encoder_ae(x1)/latent_scale
    z1_cae = encoder_cae(x1)/latent_scale
    _,z1_bvae,_ = bvae(x1)
    z1_bvae = z1_bvae/latent_scale
    z1_store = z1.detach().cpu().numpy()

    # Encode coefficient scale parameters
    log_spread0 = zeta_decoder(z0)
    coeff_spread = 5.0*(0.5*log_spread0).exp()
    zeta_use = torch.div(1.0,coeff_spread)

    x1_out = x1.permute(0,2,3,1).detach().cpu().numpy()

    # Infer the coefficients that define a path between z0 and z1
    E_single = np.zeros((num_restart_test))
    c_store = np.zeros((num_restart_test,batch_size,dict_size))
    for n in range(0,num_restart_test):
        c_data, coefficients_temp = infer_coefficients(z0.unsqueeze(-1).detach(),
                                              z1.unsqueeze(-1).detach(),
                                              transOp.get_psi(),zeta_use,
                                              device=device)
        if args.save_test:
            c_loss, steps, k_step, = c_data
        E_single[n] = c_loss
        c_store[n,:,:] = coefficients_temp.detach().cpu().numpy()
    minIdx = np.argmin(E_single)
    coefficients = torch.from_numpy(c_store[minIdx,:,:]).float().to(device)

    # Compute euclidean distance between z0 and z1 in every network
    euc_dist_store = torch.mean(F.mse_loss(z0, z1, reduce = False),1).detach().cpu().numpy()
    euc_dist_store_ae = torch.mean(F.mse_loss(z0_ae, z1_ae, reduce = False),1).detach().cpu().numpy()
    euc_dist_store_cae = torch.mean(F.mse_loss(z0_cae, z1_cae, reduce = False),1).detach().cpu().numpy()
    euc_dist_store_bvae = torch.mean(F.mse_loss(z0_bvae, z1_bvae, reduce = False),1).detach().cpu().numpy()

    # Compute coefficient sparsity
    c_sparsity_store = torch.sum(torch.abs(coefficients),1).detach().cpu().numpy()

    # Transform z0 with the inferred coefficients
    transOp.set_coefficients(coefficients)
    z1_hat = transOp(z0.unsqueeze(-1)).squeeze()
    z1_hat_store = z1_hat.detach().cpu().numpy()
    mani_offset_store = torch.mean(F.mse_loss(z1_hat, z1, reduce = False),1).detach().cpu().numpy()
    arc_len_store= compute_arc_length(transOp.get_psi(),coefficients,t_vals,z0,device = device)
    coeff_store = coefficients.detach().cpu().numpy()

    # Compute vector differences
    z_vec_change = z1_ae-z0_ae
    z_vec_change_cae = z1_cae-z0_cae
    z_vec_change_bvae = z1_bvae-z0_bvae

    # Compute interpolated and extrapolated paths for MAE
    count = 0
    for t_use in t_path_long:
        coeff_input = coefficients*t_use
        transOp.set_coefficients(coeff_input)
        z_trans_scale = transOp(z0.unsqueeze(-1)).squeeze()

        z_trans = z_trans_scale*latent_scale
        log_spread_path = zeta_decoder(z_trans_scale)
        coeff_spread = (0.5*log_spread_path).exp()

        imgTemp = decoder(z_trans)
        transImgOut = imgTemp.permute(0,2,3,1).detach().cpu().numpy()
        prob_out_img, label_est = classifier(imgTemp)
        prob_out[count,:] = prob_out_img[:,k].detach().cpu().numpy()
        _, predicted = torch.max(label_est, 1)
        acc_out[count,:] = (predicted == label).detach().cpu().numpy()
        if imgFlag == 1:
            transImgTotal[count,:,:,:,:] = transImgOut
            z_trans_out[count,:,:] = z_trans_scale.detach().cpu().numpy()
            c_spread_path[count,:,:]= coeff_spread.detach().cpu().numpy()

        count = count+1

    # Compute interpolated and extrapolated paths for AE
    count_euc = 0
    for t_use in t_path_long:
        z_vec_temp = z_vec_change*t_use
        z_euc_trans_scale = z0_ae + z_vec_temp

        z_euc_trans = z_euc_trans_scale*latent_scale
        imgTemp_euc = decoder_ae(z_euc_trans)
        transImgOut_euc = imgTemp_euc.permute(0,2,3,1).detach().cpu().numpy()
        prob_out_img_euc, label_est_euc = classifier(imgTemp_euc)
        prob_out_euc[count_euc,:] = prob_out_img_euc[:,k].detach().cpu().numpy()
        _, predicted_euc = torch.max(label_est_euc, 1)
        acc_out_euc[count_euc,:] = (predicted_euc == label_true).detach().cpu().numpy()
        if imgFlag == 1:
            transImgTotal_euc[count_euc,:,:,:,:] = transImgOut_euc
            z_trans_out_euc[count_euc,:,:] = z_euc_trans_scale.detach().cpu().numpy()
        count_euc = count_euc +1

    # Compute interpolated and extrapolated paths for CAE
    count_cae = 0
    for t_use in t_path_long:
        z_vec_temp = z_vec_change_cae*t_use
        z_cae_trans_scale = z0_cae + z_vec_temp

        z_cae_trans = z_cae_trans_scale*latent_scale
        imgTemp_cae = decoder_cae(z_cae_trans)
        transImgOut_cae = imgTemp_cae.permute(0,2,3,1).detach().cpu().numpy()
        prob_out_img_cae, label_est_cae = classifier(imgTemp_cae)
        prob_out_cae[count_cae,:] = prob_out_img_cae[:,k].detach().cpu().numpy()
        _, predicted_cae = torch.max(label_est_cae, 1)
        acc_out_cae[count_cae,:] = (predicted_cae == label).detach().cpu().numpy()
        if imgFlag == 1:
            transImgTotal_cae[count_cae,:,:,:,:] = transImgOut_cae
            z_trans_out_cae[count_cae,:,:] = z_cae_trans_scale.detach().cpu().numpy()
        count_cae= count_cae +1

    # Compute interpolated and extrapolated paths for beta-VAE
    count_bvae = 0
    for t_use in t_path_long:
        z_vec_temp = z_vec_change_bvae*t_use
        z_bvae_trans_scale = z0_bvae + z_vec_temp
        z_bvae_trans = z_bvae_trans_scale*latent_scale
        imgTemp_bvae = bvae._decode(z_bvae_trans)
        transImgOut_bvae = imgTemp_bvae.permute(0,2,3,1).detach().cpu().numpy()
        prob_out_img_bvae, label_est_bvae = classifier(imgTemp_bvae)
        prob_out_bvae[count_bvae,:] = prob_out_img_bvae[:,k].detach().cpu().numpy()
        _, predicted_bvae = torch.max(label_est_bvae, 1)
        acc_out_bvae[count_bvae,:] = (predicted_bvae == label_true).detach().cpu().numpy()
        if imgFlag == 1:
            transImgTotal_bvae[count_bvae,:,:,:,:] = transImgOut_bvae
            z_trans_out_bvae[count_bvae,:,:] = z_bvae_trans_scale.detach().cpu().numpy()
        count_bvae= count_bvae +1
    print("Class " + str(k))
    if imgFlag == 1:
        sio.savemat(test_dir + '/distTest_singleClass_' + str(k) + '.mat',{'transImgTotal':transImgTotal,'transImgTotal_euc':transImgTotal_euc,'transImgTotal_cae':transImgTotal_cae,
                    'x1':x1_out,'z0':z0_store,'z1':z1_store,'z_trans_out_euc':z_trans_out_euc,'E_restart':E_single,'prob_out':prob_out,'prob_out_euc':prob_out_euc,'prob_out_cae':prob_out_cae,
                    'z1_hat':z1_hat_store,'euc_dist':euc_dist_store,'c_sparsity':c_sparsity_store,'mani_offset':mani_offset_store,'c_spread_path':c_spread_path,'z_trans_out_cae':z_trans_out_cae,
                    'arc_len':arc_len_store,'coeff_store':coeff_store,'z_trans_out':z_trans_out,'t_path':t_path_long,'euc_dist_store_ae':euc_dist_store_ae,'euc_dist_store_cae':euc_dist_store_cae,
                    'euc_dist_store_bvae':euc_dist_store_bvae,'transImgTotal_bvae':transImgTotal_bvae,'prob_out_bvae':prob_out_bvae,'z_trans_out_bvae':z_trans_out_bvae})
    else:
        sio.savemat(test_dir + '/extrapProbTest_singleClass_' + str(k) + '.mat',{'prob_out':prob_out,'prob_out_euc':prob_out_euc,'prob_out_cae':prob_out_cae,'t_path':t_path_long,
                'acc_out':acc_out,'acc_out_euc':acc_out_euc,'acc_out_cae':acc_out_cae,'prob_out_bvae':prob_out_bvae,'acc_out_bvae':acc_out_bvae})
