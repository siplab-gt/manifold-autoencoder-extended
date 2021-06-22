"""
This script was developed for the purpose of training the manifold autoencoder with various
datasets and methods of point pair supervision. The goal is to effectively learn natural
transformations of datasets (MNIST, CIFAR-10) in the latent space of an autoencoder
using transport operators

@Filename    train_transop_MNIST.py
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
import torchvision
from torchvision import transforms

from model.transop import TransOp_expm
from model.l1_inference import infer_coefficients
from model.autoencoder import init_weights, ConvEncoder, ConvDecoder, ConvEncoder_old, ConvDecoder_old
from model.classifier import SimplifiedResNet
from model.loss import log_loss

from util.test_functions import gen_transopt_paths
from util.utils import build_vgg_nn_graph, build_nn_graph, build_labeled_vgg_nn_graph, build_label_graph
from util.dataloader import NaturalTransformationDataset, load_mnist, load_cifar10, load_celeba, load_celeba64, load_svhn, load_fmnist
from pretrain_ae import pretrain_ae


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./results/', help='folder name')
parser.add_argument('-Z', '--latent_dim', default=10, type=int, help="Dimension of latent space")
parser.add_argument('-M', '--dict_size', default=16, type=int, help="Dictionary size")
parser.add_argument('-d', '--dataset', default='mnist', type=str, help="Dataset to Use ['mnist','fmnist','svhn']")
parser.add_argument('-s', '--supervision', default='RES', type=str, help="Default supervision method for selecting point pairs")
parser.add_argument('-z', '--zeta', default=0.05, type=float, help="Zeta L1 reg")
parser.add_argument('-g', '--gamma', default=2e-5, type=float, help="Gamma L2 reg")
parser.add_argument('-plr', '--psi_lr', default=1e-3, type=float, help="Psi Learning rate")
parser.add_argument('-nlr', '--net_lr', default=1e-4, type=float, help="Net Learning rate")
parser.add_argument('-ae', '--ae_weight', default=.75, type=float, help="Scaling factor in front of the AE loss - value between 0 and 1, ae_weight + con_weight < 1.0")
parser.add_argument('-co', '--con_weight', default=.0, type=float, help="Scaling factor in front of consistency loss - value between 0 and 1, ae_weight + con_weight < 1.0")
parser.add_argument('-l', '--latent_scale', default=30.0, type=float, help="Scaling term in latent space")
parser.add_argument('-m', '--psi_var', default=.05, type=float, help="Variance to scale psi with")
parser.add_argument('-N', '--train_samples', default=50000, type=int, help="Number of training samples to use.")
parser.add_argument('-te', '--total_epochs', default=150, type=int, help="Total number of epochs.")
parser.add_argument('-nc', '--neighbor_count', default=5, type=int, help="Number of nearest neighbors to use.")
parser.add_argument('-nr', '--num_restart', default=1, type=int, help="Number of restarts for coefficient inference.")
parser.add_argument('-c', '--train_classes', default=np.arange(10), nargs='+', type=int, help="Classes to train classifier on.")
parser.add_argument('-alternate_steps_flag', type=int, default=1, help='[0/1] to specify whether to alternate between steps updating net weights and psi weights ')
parser.add_argument('-num_net_steps', type=int, default=50, help='Number of steps updating only the network and anchor weights')
parser.add_argument('-num_psi_steps', type=int, default=50, help='Number of steps updating only the psi weights')
parser.add_argument('-transfer_mod', type=int, default=4, help='Number of steps on only net updates during transfer')
parser.add_argument('-norm_flag', type=int, default=0, help='[0/1] to specify whether to normalize Psi after every gradient step')
parser.add_argument('-nae', '--norm_ae_flag', type=int, default=0, help='[0/1] to specify whether to normalize the latent space parameters')
parser.add_argument('-p', '--pretrain', action='store_true', help="Use pretrained autoencoder")
parser.add_argument('-pto', '--pretrainTO', action='store_true', help="Use pretrained AE and transport operators")
parser.add_argument('-pvgg', '--precompute', action='store_true', help="Use precomputed vgg nearest neighbors")
parser.add_argument('--TOfile', type=str, default='./', help='Transport operator file location')

# PARSE ARGUMENTS #
args = parser.parse_args()
ae_epochs = 300
total_epochs = args.total_epochs
pretrainTO = args.pretrainTO
if pretrainTO:
    pretrain = True
    transOp_epochs = 0
else:
    pretrain = args.pretrain
    transOp_epochs = 100

batch_size = 500
norm_ae_flag = args.norm_ae_flag
if norm_ae_flag == 0:
    latent_scale = args.latent_scale
else:
    latent_scale = 1.0

TOfile = args.TOfile

latent_dim = args.latent_dim
dict_size = args.dict_size
zeta = args.zeta
gamma = args.gamma
psi_var = args.psi_var
ae_weight = args.ae_weight
num_restart = args.num_restart

psi_lr = args.psi_lr
network_lr = args.net_lr

# Initialize the counts for alternating steps
num_net_steps = args.num_net_steps
num_psi_steps = args.num_psi_steps
alternate_steps_flag = args.alternate_steps_flag
net_count = 0
psi_count = num_psi_steps +1

dataset = args.dataset
train_classes = args.train_classes
train_samples = args.train_samples
neighbor_count = args.neighbor_count
supervision = args.supervision
precompute = args.precompute
max_save = np.minimum(100,batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_save = np.minimum(100,batch_size)

save_dir = args.model_path + '/' + dataset + '/' + dataset  + '_M' + str(dict_size) + '_z' + str(latent_dim) + '_zeta' + str(zeta) + '_gamma' + str(gamma) + '_plr' + str(psi_lr)  + '_nlr' + str(network_lr) + '_' + supervision
if alternate_steps_flag == 1:
    save_dir = save_dir + '_nst' + str(num_net_steps) + 'pst' + str(num_psi_steps)
if num_restart > 1:
    save_dir = save_dir + '_rest' + str(num_restart)
if args.norm_flag == 1:
    save_dir = save_dir + '_norm'
if pretrainTO:
    save_dir = save_dir + '_trainCont'
save_dir = save_dir + '/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

# Load training data
if dataset == 'mnist':
    test_imgs = 10000
    train_loader, test_loader = load_mnist('./data',batch_size, train_samples, test_imgs)
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
elif dataset == 'cifar10_vehicle':
    train_samples = None
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[0, 1, 8, 9])
    channels, image_dim, features = 3, 32, 128
    num_classes = 4
elif dataset == 'cifar10_animal':
    train_samples = None
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[3, 4, 5, 7])
    class_epochs = 0
    channels, image_dim, features = 3, 32, 128
    num_classes = 4
elif dataset == 'cifar10':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    channels, image_dim, features = 3, 32, 128
    num_classes = len(train_classes)
elif dataset == 'svhn':
    train_loader, test_loader = load_svhn('./data', batch_size, train_samples =train_samples,train_classes=train_classes)
    channels, image_dim, features = 3, 32, 128
    num_classes = len(train_classes)
elif dataset == 'fmnist':
    train_loader, test_loader = load_fmnist('./data', batch_size, train_classes=train_classes)
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
elif dataset == 'celeba':
    train_loader, test_loader = load_celeba('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    channels, image_dim, features = 3, 32, 128
    num_classes = len(train_classes)
elif dataset == 'celeba64':
    train_loader, test_loader = load_celeba64('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    channels, image_dim, features = 3, 64, 128
    num_classes = len(train_classes)

# Select point pair supervision strategy
if supervision == 'RES':
    if precompute:
        checkpoint = torch.load(args.model_path + 'pretrained/resnet_nn_' + dataset + '.pt')
        nearest_neighbor = checkpoint['nearest_neighbor']
    else:
        res = models.resnet18(pretrained=True).to(device)
        simplified_res = SimplifiedResNet(res, image_dim=image_dim).to(device)
        simplified_res.eval()
        nearest_neighbor = build_vgg_nn_graph(train_loader, latent_dim,
                                              simplified_res,neighbor_count, device)
        torch.save({
            'nearest_neighbor': nearest_neighbor,
        }, args.model_path + 'pretrained/resnet_nn_{}.pt'.format(dataset))
    if type(train_loader.dataset) is NaturalTransformationDataset:
        train_loader.dataset.set_nn_graph(nearest_neighbor)
    else:
        train_loader.dataset.dataset.set_nn_graph(nearest_neighbor)
    print("Using classifier embedding nearest neighbor for point pair supervision.")
elif supervision == 'LRES':
    if precompute:
        checkpoint = torch.load(args.model_path + 'pretrained/lresnet_nn_' + dataset + '.pt')
        nearest_neighbor = checkpoint['nearest_neighbor']
    else:
        res = models.resnet18(pretrained=True).to(device)
        simplified_res = SimplifiedResNet(res, image_dim=image_dim).to(device)
        simplified_res.eval()
        nearest_neighbor = build_labeled_vgg_nn_graph(train_loader, latent_dim,
                                                      simplified_vgg, num_classes, neighbor_count, device)
        torch.save({
            'nearest_neighbor': nearest_neighbor,
        }, args.model_path + 'pretrained/lresnet_nn_{}.pt'.format(dataset))
    if type(train_loader.dataset) is NaturalTransformationDataset:
        train_loader.dataset.set_nn_graph(nearest_neighbor)
    else:
        train_loader.dataset.dataset.set_nn_graph(nearest_neighbor)
    print("Using classifier embedding nearest neighbor within a labeled class for point pair supervision.")
elif supervision == 'LABEL':
    nearest_neighbor = build_label_graph(train_loader, latent_dim, simplified_vgg, num_classes, neighbor_count, device)
    if type(train_loader.dataset) is NaturalTransformationDataset:
        train_loader.dataset.set_nn_graph(nearest_neighbor)
    else:
        train_loader.dataset.dataset.set_nn_graph(nearest_neighbor)
    print("Using random samples from the same class for point pair supervision.")
else:
    print("Using encoder nearest neighbor for point pair supervision.")

# Initialize autoencoder model
encoder = ConvEncoder(latent_dim, channels, image_dim, norm_ae_flag, num_filters=features).to(device)
decoder = ConvDecoder(latent_dim, channels, image_dim, num_filters=features).to(device)
encoder.apply(init_weights)
decoder.apply(init_weights)
autoenc_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=network_lr, betas=(0.5, 0.999))
ae_scheduler = torch.optim.lr_scheduler.ExponentialLR(autoenc_opt, gamma=0.985)

# Initialize transport operator model
transOp = TransOp_expm(M=dict_size, N=latent_dim, var=psi_var).to(device)

transOp_opt = torch.optim.Adam(transOp.parameters(), lr=psi_lr, weight_decay=gamma)
scheduler = torch.optim.lr_scheduler.ExponentialLR(transOp_opt, gamma=0.985)

# Pre-train autoencoder or load a pre-trained one
if pretrain:
    model_state = torch.load(args.model_path + 'pretrained/pretrain_{}_ae_Z{}.pt'.format(dataset, latent_dim))
    encoder.load_state_dict(model_state['encoder'])
    decoder.load_state_dict(model_state['decoder'])
    print("Successfully loaded pre-trained autoencoder.")
else:
    # Pre-train AE
    autoenc_scheduler = torch.optim.lr_scheduler.ExponentialLR(autoenc_opt, gamma=0.995)
    pretrain_ae(encoder, decoder, train_loader, test_loader, autoenc_opt,
                autoenc_scheduler, dataset, latent_dim, device, ae_epochs)
if pretrainTO:
    modelTO_state = torch.load(TOfile)
    transOp.load_state_dict(modelTO_state['transOp'])
    encoder.load_state_dict(modelTO_state['encoder'])
    decoder.load_state_dict(modelTO_state['decoder'])
    print('Successfully loaded transport operator model')

total_ae_loss = np.zeros(total_epochs*len(train_loader))
total_transOp_loss = np.zeros(total_epochs*len(train_loader))

total_latent_mag = np.zeros(total_epochs*len(train_loader))
transObj_diff = np.zeros(total_epochs*len(train_loader))
total_nz_coeff = np.zeros((total_epochs*len(train_loader), dict_size+1))
total_dict_mag = np.zeros((total_epochs*len(train_loader), dict_size))
epoch_time = np.zeros(total_epochs*len(train_loader))

counter = 0
# Transport Operator training
for j in range(total_epochs):

    if supervision == 'NN':
        nearest_neighbor = build_nn_graph(train_loader, latent_dim, encoder, device=device)
        train_loader.dataset.dataset.set_nn_graph(nearest_neighbor)

    avg_transOp_loss = torch.zeros(len(train_loader))
    avg_ae_loss = torch.zeros(len(train_loader))

    for idx, batch in enumerate(train_loader):
        # Draw next batch
        x0, x1, labels = batch
        pre_time = time.time()

        # Move vector to device memory
        x0, x1 = x0.to(device), x1.to(device)

        # Inference stability
        z0 = encoder(x0)/latent_scale
        z1 = encoder(x1)/latent_scale

        # Feed ground truth images through decoder for reconstruction loss
        x0_hat = decoder(z0*latent_scale)
        x1_hat = decoder(z1*latent_scale)

        transOp_opt.zero_grad()
        autoenc_opt.zero_grad()

        ae_loss = 0.5*F.mse_loss(x0_hat, x0, reduction='mean') + 0.5*F.mse_loss(x1_hat, x1, reduction='mean')

        # Transport operator training phase OR 3/4 of the fine-tuning steps (dictated by transfer mod)
        if (j <= transOp_epochs) or (np.mod(counter,args.transfer_mod) != 0):
            # Infer coefficients between latent points to infer paths
            E_single = np.zeros((num_restart))
            c_store = np.zeros((num_restart,batch_size,dict_size))
            for m in range(0,num_restart):
                c_data, coefficients_temp = infer_coefficients(z0.unsqueeze(-1).detach(),
                                                      z1.unsqueeze(-1).detach(),
                                                      transOp.get_psi(), zeta,
                                                      device=device)
                c_loss, steps, k, = c_data
                E_single[m] = c_loss
                c_store[m,:,:] = coefficients_temp.detach().cpu().numpy()
            minIdx = np.argmin(E_single)
            coefficients = torch.from_numpy(c_store[minIdx,:,:]).float().to(device)

            # Use inferred coefficients to estimate latent path
            transOp.set_coefficients(coefficients)
            z1_hat = transOp(z0.unsqueeze(-1)).squeeze()
            transOp_loss = F.mse_loss(z1_hat, z1, reduction='mean')

        # Fine-tuning training
        if j > transOp_epochs:
            if np.mod(counter,args.transfer_mod) != 0:
                loss_use = (1-ae_weight)*transOp_loss + ae_weight*ae_loss
                if (net_count < num_net_steps or alternate_steps_flag == 0):
                    loss_use.backward()
                    autoenc_opt.step()
                    net_count = net_count + 1
                if (psi_count < num_psi_steps or alternate_steps_flag == 0):
                    if alternate_steps_flag == 1:
                        loss_use.backward()
                    transOp_opt.step()
                    psi_count = psi_count +1
            else:
                ae_loss.backward()
                autoenc_opt.step()

        # Only training transport operators
        else:
            transOp_loss.backward()
            transOp_opt.step()

        if net_count == num_net_steps and alternate_steps_flag == 1:
            net_count = net_count+1
            psi_count = 0
        if psi_count == num_psi_steps and alternate_steps_flag == 1:
            psi_count = psi_count +1
            net_count = 0

        if args.norm_flag == 1:
            normalize_val = 0.5
            psi_new= transOp.get_psi()
            psi_norm = torch.diag(torch.div(normalize_val,torch.sqrt(torch.sum(torch.square(torch.reshape(psi_new,(dict_size,latent_dim*latent_dim))),axis = 1))))
            psi_new_norm = torch.matmul(torch.transpose(torch.reshape(psi_new,(dict_size,latent_dim*latent_dim)),0,1),psi_norm)
            psi_square = torch.reshape(torch.transpose(psi_new_norm,0,1),(dict_size,latent_dim,latent_dim))
            transOp.set_psi(psi_square)

        # Save information from training iteration
        avg_ae_loss[idx] = ae_loss.item()
        total_ae_loss[j*len(train_loader) + idx] = ae_loss.item()
        total_latent_mag[j*len(train_loader) + idx] = z0.norm(dim=1).mean().detach().cpu().numpy()
        epoch_time[counter] =  time.time() - pre_time
        if (j <= transOp_epochs) or (np.mod(counter,args.transfer_mod) != 0):
            avg_transOp_loss[idx] = transOp_loss.item()
            count_nz = np.zeros(dict_size+1, dtype=int)
            coeff_np = coefficients.detach().cpu().numpy()
            total_nz = np.count_nonzero(coeff_np, axis=1)
            for z in range(len(total_nz)):
                count_nz[total_nz[z]] += 1
            psi_norm = transOp.psi.reshape(dict_size, -1).norm(dim=1)
            total_transOp_loss[j*len(train_loader) + idx] = transOp_loss.item()
            total_nz_coeff[j*len(train_loader) + idx] = count_nz
            total_dict_mag[j*len(train_loader) + idx] = psi_norm.detach().cpu().numpy()
            transOp.set_coefficients(coefficients)
            post_z1_hat = transOp(z0.unsqueeze(-1)).squeeze()
            post_loss = F.mse_loss(post_z1_hat,z1, reduction='mean').item()
            transObj_diff[j*len(train_loader) + idx] = transOp_loss.item() - post_loss
        if np.mod(counter,args.transfer_mod) == 0 and counter > 0:
            total_dict_mag[j*len(train_loader) + idx] = total_dict_mag[j*len(train_loader) + idx-1]
        print("[Epoch %d/%d] [Batch %d/%d] [Loss AE: %4.8f] [PreLoss: %4.8f] [LossDiff: %.2e] [Time: %2.2f] sec" % (
            j+1, total_epochs,idx+1, len(train_loader),ae_loss.item(), transOp_loss.item()*(1-ae_weight),transObj_diff[j*len(train_loader) + idx], epoch_time[counter]))

        counter = counter + 1
    print("Epoch {} of {}: TransOp Loss: {:.4E}\t AE Loss: {:.3E}"
          .format(j+1, total_epochs, avg_transOp_loss.mean(), avg_ae_loss.mean(),))

    # Decay step-size for transport operators
    if j > transOp_epochs:
        scheduler.step()
        ae_scheduler.step()

    if (np.mod(j+1,4) == 0 or j+1 == total_epochs or j == transOp_epochs):
        if j == transOp_epochs:
            saveName = save_dir + 'modelDict_{}_M{}Z{}zeta{}gam{}_transOpTrain.pt'.format(dataset, dict_size,latent_dim,zeta,gamma)
        else:
            saveName = save_dir + 'modelDict_{}_M{}Z{}_step{}.pt'.format(dataset, dict_size, latent_dim,counter)
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'transOp': transOp.state_dict(),
            'transOp_opt':transOp_opt.state_dict(),
            'autoenc_opt':autoenc_opt.state_dict(),
            'supervision': supervision,
            'ae_loss': total_ae_loss,
            'transOp_loss': total_transOp_loss,
            'latent_scale': latent_scale,
            'latent_mag': total_latent_mag,
            'nz_coeff': total_nz_coeff,
            'dict_mag': total_dict_mag,
            'zeta': zeta,
            'gamma': gamma,
            'train_samples': train_samples,
            'ae_weight': ae_weight,
        }, saveName)

    x1_est = decoder(z1_hat.squeeze()*latent_scale)
    num_plots = 6
    fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(8, 12))
    for p in range(num_plots):
        ax[p, 0].imshow(x0[p].detach().cpu().permute(1, 2, 0).squeeze().numpy().clip(0, 1))
        ax[p, 1].imshow(x1[p].detach().cpu().permute(1, 2, 0).squeeze().numpy().clip(0, 1))
        ax[p, 2].imshow(x1_hat[p].detach().cpu().permute(1, 2, 0).squeeze().numpy().clip(0, 1))
        ax[p, 3].imshow(x1_est[p].detach().cpu().permute(1, 2, 0).squeeze().numpy().clip(0, 1))
    ax[0,0].set_title('x0')
    ax[0,1].set_title('x1')
    ax[0,2].set_title('z1 Dec')
    ax[0,3].set_title('z1_hat Dec')
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.savefig(save_dir + '{}_M{}Z{}_step{}_transforms.png'.format(
                dataset, dict_size, latent_dim,counter),
                bbox_inches='tight')
    plt.close()
    if (np.mod(j+1,4) == 0 or j+1 == total_epochs or j == transOp_epochs or j+1 == 1):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax1 = plt.subplot(121)
        ax1.plot(total_ae_loss[:counter])
        ax1.set_title('Reconstruction Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Reconstruction Loss')
        ax2 = plt.subplot(122)
        ax2.plot(total_transOp_loss[:counter])
        ax2.set_title('Transport Operator Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Transport Operator Loss')
        plt.savefig(save_dir + 'lossVals_{}_M{}Z{}.png'.format(
                dataset, dict_size, latent_dim),
                bbox_inches='tight')
        plt.close()
        plt.scatter(range(0,counter),transObj_diff[:counter],c=np.sign(transObj_diff[:counter]) )
        plt.ylim(-5*10^-6, 100*10^-6)
        plt.xlabel('Training Step')
        plt.ylabel('Diff in Trans Obj After Step')
        plt.savefig(save_dir + 'transOptDiff_{}_M{}Z{}.png'.format(
                dataset, dict_size, latent_dim),
                bbox_inches='tight')
        plt.close()
        plt.imshow(np.absolute(coefficients.detach().cpu().numpy()[:max_save]))
        plt.clim(0,0.01)
        plt.xlabel('Transport Operator Number')
        plt.ylabel('Sample Number')
        plt.savefig(save_dir + 'coeffVal_{}_M{}Z{}_step{}.png'.format(
                dataset, dict_size, latent_dim,counter),
                bbox_inches='tight')
        plt.close()
        plt.plot(total_dict_mag[:counter,:])
        plt.xlabel('Training Step')
        plt.ylabel('Dictionary Magnitudes');
        plt.savefig(save_dir + 'dictMag_{}_M{}Z{}.png'.format(
                dataset, dict_size, latent_dim),
                bbox_inches='tight')
        plt.close()
        coeff_range = 5
        train_example = 5
        #z = z0[5].unsqueeze(-1).to(device)
        coeff_interp = np.linspace(-coeff_range, coeff_range, 100)
        fig, ax = plt.subplots(nrows=dict_size//3 + 1, ncols=3, figsize=(16, 18))
        plt.subplots_adjust(hspace=0.4, top=.9)
        for i in range(dict_size):
            row = int(i / 3)
            column = int(i % 3)
            z_path = []
            coeff_zero = np.zeros((batch_size,dict_size))
            for coeff_use in coeff_interp:
                coeff_zero[:,i] = coeff_use
                coeff_use = torch.from_numpy(coeff_zero).float().to(device)
                transOp.set_coefficients(coeff_use)
                z1_hat = transOp(z0.unsqueeze(-1)).squeeze()[train_example]
                z_path.append(z1_hat)
            z_path = torch.stack(z_path)
            for z_dim in range(latent_dim):
                ax[row, column].plot(coeff_interp, z_path[:, z_dim].detach().cpu().numpy())
                ax[row, column].set_ylim((-3,3))
        plt.savefig(save_dir + 'transOpGen_{}_M{}Z{}_step{}.png'.format(
                dataset, dict_size, latent_dim,counter),
                bbox_inches='tight')
        plt.close()
