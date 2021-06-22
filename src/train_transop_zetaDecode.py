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

from model.transop import TransOp_expm, ZetaDecoder
from model.l1_inference import infer_coefficients
from model.autoencoder import init_weights, ConvEncoder, ConvDecoder, ConvEncoder_old, ConvDecoder_old
from model.classifier import SimplifiedVGG
from model.loss import log_loss, l2_loss, kld_loss
from model.sampler import sample_c
from util.utils import build_vgg_nn_graph, build_nn_graph, print_statistics
from util.dataloader import load_mnist, load_cifar10, load_svhn, load_fmnist
from util.test_functions import compute_cSpread

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../results/', help='folder name')
parser.add_argument('-Z', '--latent_dim', default=32, type=int, help="Dimension of latent space")
parser.add_argument('-M', '--dict_size', default=30, type=int, help="Dictionary size")
parser.add_argument('-d', '--dataset', default='cifar10', type=str, help="Dataset to Use")
parser.add_argument('-s', '--supervision', default='VGG', type=str, help="Default supervision method for selecting point pairs")
parser.add_argument('-z', '--zeta', default=0.5, type=float, help="Zeta L1 reg")
parser.add_argument('-g', '--gamma', default=2e-6, type=float, help="Gamma L2 reg")
parser.add_argument('-plr', '--psi_lr', default=1e-3, type=float, help="Psi Learning rate")
parser.add_argument('-nlr', '--net_lr', default=1e-4, type=float, help="Net Learning rate")
parser.add_argument('-ae', '--ae_weight', default=.75, type=float, help="Scaling factor in front of the AE loss - value between 0 and 1")
parser.add_argument('-l', '--latent_scale', default=3.0, type=int, help="Scaling term in latent space")
parser.add_argument('-m', '--psi_var', default=.05, type=float, help="Variance to scale psi with")
parser.add_argument('-N', '--train_samples', default=50000, type=int, help="Number of training samples to use.")
parser.add_argument('-Np', '--train_sample_pseudo', default=50000, type=int, help="Number of training samples to use to train pseudolabels.")
parser.add_argument('-nc', '--neighbor_count', default=5, type=int, help="Number of nearest neighbors to use.")
parser.add_argument('-c', '--train_classes', default=np.arange(10), nargs='+', type=int, help="Classes to train classifier on.")
parser.add_argument('-nae', '--norm_ae_flag', type=int, default=0, help='[0/1] to specify whether to normalize the latent space parameters')
parser.add_argument('-zkl', '--kl_zeta', type=float, default=0.1, help='Weight to use on the KL term to use on the coefficie. Set to 0 to use non.')
parser.add_argument('-pkl', '--kl_prior', type=float, default=0.1, help='The spread of the prior on the laplace distribution')
parser.add_argument('-css', '--coeff_spread_scale', type=float, default=0.1, help='Scale on coefficient spread weights')
parser.add_argument('-ct', '--classifier_type', default='mlp', type=str, help="[cnn,resnet,mlp,lenet] Specify which classifier to use")
parser.add_argument('-cd', '--classifier_domain', default='latent', type=str, help="[image,latent] Specify whether to train the classifier on the latent vectors or images")
parser.add_argument('-cs', '--classifier_steps', default='net', type=str, help="[net,psi] Specify whether to use consistency loss with net update steps or psi update steps")
parser.add_argument('-cl', '--con_loss', default='kld', type=str, help="[log,l2,kld] Determine which consistency regularizer loss to use")
parser.add_argument('-cae', '--con_pretrain', default=0, type=int, help="Flag specifying whether a classifier was used during ae pretraining")
parser.add_argument('-r', '--run', required=True, type=int, help="Run number")
parser.add_argument('-p', '--pretrain', action='store_true', help="Use pretrained autoencoder")
parser.add_argument('-pclf', '--pretrainclf', action='store_true', help="Use pretrained clasasifier")
parser.add_argument('-pvgg', '--precompute', action='store_true', help="Use precomputed vgg nearest neighbors")
parser.add_argument('-v', '--verbose', action='store_true', help="Verbosity 1")
parser.add_argument('-V', '--vverbose', action='store_true', help="Verbosity 2")
parser.add_argument('--nonneg', action='store_true', help="Non-negative constraint on coefficients.")
parser.add_argument('-st', '--save_test', action='store_true', help="Save test data")
parser.add_argument('--TOfile',required=True, type=str, help='Transport operator file location')

# PARSE ARGUMENTS #
args = parser.parse_args()
total_epochs = 300
class_epochs = 150
pretrainclf = args.pretrainclf



TOfile = args.TOfile


batch_size = 250
norm_ae_flag = args.norm_ae_flag
if norm_ae_flag == 0:
    latent_scale = args.latent_scale
else:
    latent_scale = 1.0


latent_dim = args.latent_dim
dict_size = args.dict_size
zeta = args.zeta
gamma = args.gamma
psi_var = args.psi_var
non_neg = args.nonneg
zeta_kl = args.kl_zeta
prior_kl = args.kl_prior
coeff_spread_scale = args.coeff_spread_scale
train_sample_pseudo = args.train_sample_pseudo
con_pretrain = args.con_pretrain
con_loss = args.con_loss

psi_lr = args.psi_lr
network_lr = args.net_lr



dataset = args.dataset
train_classes = args.train_classes
train_samples = args.train_samples
neighbor_count = args.neighbor_count
supervision = args.supervision
precompute = args.precompute
run_number = args.run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

verbose = 0
if args.verbose:
    verbose = 1
if args.vverbose:
    verbose = 2


save_dir = args.model_path + '/' + dataset + '/' + dataset  + '_zetaNet_M' + str(dict_size) + '_z' + str(latent_dim) + '_zeta' + str(zeta) + '_gamma' + str(gamma) + '_plr' + str(psi_lr)  + '_nlr' + str(network_lr)+ '_' + args.classifier_type + '_'  + args.classifier_domain + '_' + args.classifier_steps  + '_' + supervision + '_' + con_loss

if zeta_kl > 0.0:
    save_dir = save_dir + '_zkl' + str(zeta_kl) + '_zprior' + str(prior_kl)
class_trans_str = np.array2string(train_classes)
if con_pretrain == 1:
    save_dir = save_dir + '_wCon'
test_dir = save_dir + '_zetaOnly_fix_test/'
save_dir = save_dir + '_zetaOnly_fix/'




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
    train_loader_pseudo, _ = load_mnist('./data',batch_size, train_sample_pseudo, test_imgs)
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


if supervision == 'VGG':
    if precompute:
        checkpoint = torch.load('../results/pretrained/vgg_nn_' + dataset + '.pt')
        nearest_neighbor = checkpoint['nearest_neighbor']
    else:
        vgg = models.resnet18(pretrained=True).to(device)
        simplified_vgg = SimplifiedVGG(vgg).to(device)
        simplified_vgg.eval()
        nearest_neighbor = build_vgg_nn_graph(train_loader, latent_dim, simplified_vgg,neighbor_count, device)
        torch.save({
            'nearest_neighbor': nearest_neighbor,
        }, '../results/pretrained/vgg_nn_{}.pt'.format(dataset))


    train_loader.dataset.dataset.set_nn_graph(nearest_neighbor)
    print("Using claissifier embedding nearest neighbor for point pair supervision.")
elif supervision == 'NN':
    print("Using encoder nearest neighbor for point pair supervision.")

# Initialize autoencoder model
if dataset == 'svhn':
    encoder = ConvEncoder_old(latent_dim, channels, image_dim, norm_ae_flag, num_filters=features).to(device)
    decoder = ConvDecoder_old(latent_dim, channels, image_dim, num_filters=features).to(device)
else:
    encoder = ConvEncoder(latent_dim, channels, image_dim, norm_ae_flag, num_filters=features).to(device)
    decoder = ConvDecoder(latent_dim, channels, image_dim, num_filters=features).to(device)
encoder.apply(init_weights)
decoder.apply(init_weights)
autoenc_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=network_lr, betas=(0.5, 0.999))
if args.classifier_type == 'mlp' or args.classifier_domain == 'latent':
    from model.mlp import Classifier
    if args.classifier_domain == 'latent':
        classifier = Classifier(len(train_classes),latent_dim).to(device)
    else:
        classifier = Classifier(len(train_classes),image_dim*image_dim*channels).to(device)
elif args.classifier_type == 'cnn':
    from model.classifier import CNN
    classifier = CNN(len(train_classes)).to(device)
elif args.classifier_type == 'lenet':
    from model.classifier import LeNet
    classifier = LeNet(len(train_classes)).to(device)
elif args.classifier_type == 'resnet':
    from model.resnet import ResNet, BasicBlock
    classifier = ResNet(BasicBlock, [3, 3, 3], num_classes=len(train_classes)).to(device)
elif args.classifier_type == 'resnet18':
    from model.resnet18 import ResNet18
    classifier = ResNet18().to(device)
    classifier = torch.nn.DataParallel(classifier)
#ae_scheduler = torch.optim.lr_scheduler.ExponentialLR(autoenc_opt, gamma=0.995)

transOp = TransOp_expm(M=dict_size, N=latent_dim, var=psi_var).to(device)


modelTO_state = torch.load(TOfile)
transOp.load_state_dict(modelTO_state['transOp'])
encoder.load_state_dict(modelTO_state['encoder'])
decoder.load_state_dict(modelTO_state['decoder'])
#transOp_opt.load_state_dict(modelTO_state['transOp_opt'])
#autoenc_opt.load_state_dict(modelTO_state['autoenc_opt'])
if args.classifier_domain == 'latent' and pretrainclf:
    classifier.load_state_dict(modelTO_state['FTclassifier'])
elif args.classifier_domain == 'image' and pretrainclf:
    model_state = torch.load('../results/pretrained/pretrain_classifier_{}_{}_{}_ae_Z{}.pt'.format(dataset, args.classifier_type,args.classifier_domain, latent_dim))
    classifier.load_state_dict(model_state['classifier'])
print('Successfully loaded transport operator model')


if not pretrainclf:
    clf_opt = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    if args.classifier_type == 'resnet':
        clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(clf_opt, T_max=200)
    else:
        clf_scheduler = torch.optim.lr_scheduler.ExponentialLR(clf_opt, gamma=0.995)
    for j in range(class_epochs):

        for idx, batch in enumerate(train_loader):
            x0, _, label = batch
            label = label.to(device)
            x0 = x0.clamp(0, 1).to(device)
            clf_opt.zero_grad()
            z0 = encoder(x0)

            if args.classifier_domain == 'latent':
                prob_out, label_est = classifier(z0)
            else:
                prob_out, label_est = classifier(x0)

            clf_loss = F.cross_entropy(label_est, label)
            clf_loss.backward()
            clf_opt.step()

        test_error = torch.zeros(len(test_loader))
        with torch.no_grad():

            accuracy = 0
            ce_loss = torch.zeros(len(test_loader))
            for idx, batch in enumerate(test_loader):
                x0, _,label = batch
                x0 = x0.clamp(0, 1).to(device)
                label = label.to(device)
                z0 = encoder(x0)

                if args.classifier_domain == 'latent':
                    prob_out, label_est = classifier(z0)
                else:
                    prob_out, label_est = classifier(x0)
                _, predicted = torch.max(label_est, 1)
                accuracy += (predicted == label).sum().item()

                ce_loss[idx] = F.cross_entropy(label_est, label)
        accuracy = accuracy / len(test_loader.dataset)
        print(f"Epoch {j+1} of 200 -- CE Loss: {ce_loss.mean():.2E}, Accuracy: {100*accuracy:.2f}")

        if j > 150 or args.classifier_type == 'resnet':
            clf_scheduler.step()

    if args.classifier_domain == 'latent':
        modelTO_state['transOp'] = transOp.state_dict()
        modelTO_state['encoder'] = encoder.state_dict()
        modelTO_state['decoder'] = decoder.state_dict()
        modelTO_state['FTclassifier'] = classifier.state_dict()
        torch.save(modelTO_state, TOfile)


    else:
        saveDict = {
            'classifier':classifier.state_dict(),
        }
        torch.save(saveDict, '../results/pretrained/pretrain_classifier_{}_{}_{}_ae_Z{}.pt'.format(dataset, args.classifier_type,args.classifier_domain, latent_dim))


zeta_decoder = ZetaDecoder(latent_dim, dict_size).to(device)
zeta_opt = torch.optim.Adam(zeta_decoder.parameters(), lr=1e-3)
zeta_scheduler = torch.optim.lr_scheduler.ExponentialLR(zeta_opt, gamma=0.985)

total_con0_loss = np.zeros(total_epochs*len(train_loader))
total_con1_loss = np.zeros(total_epochs*len(train_loader))
total_kl_loss = np.zeros(total_epochs*len(train_loader))

epoch_time = np.zeros(total_epochs*len(train_loader))
loss_con_0 = torch.tensor([-1])
loss_con_1 = torch.tensor([-1])

counter = 0
# Transport Operator training
for j in range(total_epochs):

    if supervision == 'NN':
        nearest_neighbor = build_nn_graph(train_loader, latent_dim, encoder, 5, device)
        train_loader.dataset.dataset.set_nn_graph(nearest_neighbor)


    for idx, batch in enumerate(train_loader):
        # Draw next batch

        x0, x1, labels = batch
        pre_time = time.time()
        # Use supervision to find point pair
        if supervision == 'GCE':
            x0 = x0.clamp(0, 1).to(device)
            # Find GCE latent vector
            gce_latent, _, _ = gce_encoder(x0)
            # Pick a random causal dimension to perturb
            causal_dimension = np.random.randint(K, K+L, size=batch_size)
            latent_add = torch.zeros((batch_size,K+L), device=device)
            # For each batch element, perturn causal element
            for change_idx in range(batch_size):
                latent_add[change_idx, causal_dimension[change_idx]] = torch.rand(1, device=device)*2 - 1
            # Decode original image and causally perturbed to get point pair
            x0 = gce_decoder(gce_latent)
            x1 = gce_decoder(gce_latent + latent_add)

        # Move vector to device memory
        x0, x1 = x0.clamp(0, 1).to(device), x1.clamp(0, 1).to(device)


        zeta_opt.zero_grad()
        z0_con = encoder(x0)/latent_scale
        z1_con = encoder(x1) /latent_scale
        # Use classifier to shape zeta posterior for principled sampling
        log_spread0 = zeta_decoder(z0_con)
        coeff_spread0 = coeff_spread_scale*(0.5*log_spread0).exp()
        coeff_samp_0 = sample_c(batch_size,dict_size,coeff_spread0)
        transOp.set_coefficients(coeff_samp_0)
        z0_samp = transOp(z0_con.unsqueeze(-1)).squeeze()
        x0_samp = decoder(z0_samp*latent_scale )
        x0_hat = decoder(z0_con*latent_scale)

        log_spread1 = zeta_decoder(z1_con)
        coeff_spread1 =coeff_spread_scale*(0.5*log_spread1).exp()
        coeff_samp_1 = sample_c(batch_size,dict_size,coeff_spread1)
        transOp.set_coefficients(coeff_samp_1)
        z1_samp = transOp(z1_con.unsqueeze(-1)).squeeze()
        x1_samp = decoder(z1_samp*latent_scale )

        if args.classifier_domain == 'latent':
            prob_output0, output0 = classifier(z0_con*latent_scale)
            prob_output_samp0, output_samp0 = classifier(z0_samp*latent_scale)
            prob_output1, output1 = classifier(z1_con*latent_scale)
            prob_output_samp1, output_samp1 = classifier(z1_samp*latent_scale)
        else:
            x0_hat_con = decoder(z0_con*latent_scale)
            x1_hat_con = decoder(z1_con*latent_scale)
            prob_output0, output0 = classifier(x0_hat_con)
            prob_output_samp0, output_samp0 = classifier(x0_samp)
            prob_output1, output1 = classifier(x1_hat_con)
            prob_output_samp1, output_samp1 = classifier(x1_samp)
        if con_loss == 'log':
            loss_con_0 = log_loss(output0, prob_output0, output_samp0,prob_output_samp0)
            loss_con_1 = log_loss(output1, prob_output1, output_samp1,prob_output_samp1)
        elif con_loss == 'l2':
            loss_con_0 = l2_loss(output0, prob_output0, output_samp0,prob_output_samp0)
            loss_con_1 = l2_loss(output1, prob_output1, output_samp1,prob_output_samp1)
        elif con_loss == 'kld':
            loss_con_0 = kld_loss(output0, prob_output0, output_samp0,prob_output_samp0)
            loss_con_1 = kld_loss(output1, prob_output1, output_samp1,prob_output_samp1)
        # Take steps on zeta decoder
        zeta_loss = 1/2.0*(loss_con_0+ loss_con_1)
        if zeta_kl > 1e-6:
            #kl_loss = 0.5*zeta_kl*(-log_spread0 + np.log(prior_kl) + log_spread0.exp()/prior_kl-1.0).mean()
            #kl_loss += 0.5*zeta_kl*(-log_spread1 + np.log(prior_kl) + log_spread1.exp()/prior_kl-1.0).mean()
            #kl_loss = 0.5*zeta_kl*(log_spread0 - np.log(prior_kl) + prior_kl/log_spread0.exp()-1.0).mean()
            #kl_loss += 0.5*zeta_kl*(log_spread1 - np.log(prior_kl) + prior_kl/log_spread1.exp()-1.0).mean()
            kl_loss = 0.5*zeta_kl*(-torch.log(coeff_spread0) + np.log(prior_kl) + coeff_spread0/prior_kl-1.0).mean()
            kl_loss += 0.5*zeta_kl*(-torch.log(coeff_spread1) + np.log(prior_kl) + coeff_spread1/prior_kl-1.0).mean()
            total_kl_loss[counter] = kl_loss.item()
            zeta_loss += kl_loss
        zeta_loss.backward()
        zeta_opt.step()

        total_con0_loss[j*len(train_loader) + idx] = loss_con_0.item()
        total_con1_loss[j*len(train_loader) + idx] = loss_con_1.item()
        print("[Epoch %d/%d] [Batch %d/%d] [Time: %2.2f] sec" % (
            j+1, total_epochs,idx+1, len(train_loader), epoch_time[counter]))
        #if args.save_test and (np.mod(counter,2) == 0 and counter < 28):
        #    psi_save = transOp.get_psi().detach().cpu().numpy()
        #    x0_img = x0[0:16,:,:,:].permute(0,2,3,1).detach().cpu().numpy()
        #    saveName = test_dir + 'psiVal_step' + str(counter) + '.mat'
        #    sio.savemat(saveName,{'Psi':psi_save,'epoch_time':epoch_time[:counter],
        #                          'z0':z0_con.detach().cpu().numpy()[0:100],'z1':z1_con.detach().cpu().numpy()[0:100],'x0':x0_img[0:20],'params':args,
        #                          'batch_labels':labels.detach().cpu().numpy(),'prob_output0':prob_output0.detach().cpu().numpy()[0:100],'prob_output1':prob_output1.detach().cpu().numpy()[0:100],'prob_output_samp0':prob_output_samp0.detach().cpu().numpy()[0:100],
        #                          'prob_output_samp1':prob_output_samp1.detach().cpu().numpy()[0:100],'con_loss_0':total_con0_loss[:counter],'con_loss_1':total_con1_loss[:counter],
        #                          'coeff_samp_0':coeff_samp_0.detach().cpu().numpy()[0:100],'coeff_samp_1':coeff_samp_1.detach().cpu().numpy()[0:100],'x0_samp':x0_samp.detach().cpu().numpy()[0:20],
        #                          'x0_hat':x0_hat.detach().cpu().numpy()[0:20],'z0_samp':z0_samp.detach().cpu().numpy()[0:100],'z1_samp':z1_samp.detach().cpu().numpy()[0:100],
        #                          'coeff_spread0':coeff_spread0.detach().cpu().numpy()[0:100],'coeff_spread1':coeff_spread1.detach().cpu().numpy()[0:100],'total_kl_loss':total_kl_loss[:counter]})
        counter = counter + 1



    zeta_scheduler.step()


    if args.save_test and (np.mod(j+1,50) == 0 or j+1 == total_epochs or j < 3):
        psi_save = transOp.get_psi().detach().cpu().numpy()
        x0_img = x0[0:16,:,:,:].permute(0,2,3,1).detach().cpu().numpy()
        saveName = test_dir + 'psiVal_step' + str(counter) + '.mat'
        sio.savemat(saveName,{'Psi':psi_save,'epoch_time':epoch_time[:counter],
                              'z0':z0_con.detach().cpu().numpy()[0:100],'z1':z1_con.detach().cpu().numpy()[0:100],'x0':x0_img[0:20],'params':args,
                              'batch_labels':labels.detach().cpu().numpy(),'prob_output0':prob_output0.detach().cpu().numpy()[0:100],'prob_output1':prob_output1.detach().cpu().numpy()[0:100],'prob_output_samp0':prob_output_samp0.detach().cpu().numpy()[0:100],
                              'prob_output_samp1':prob_output_samp1.detach().cpu().numpy()[0:100],'con_loss_0':total_con0_loss[:counter],'con_loss_1':total_con1_loss[:counter],
                              'coeff_samp_0':coeff_samp_0.detach().cpu().numpy()[0:100],'coeff_samp_1':coeff_samp_1.detach().cpu().numpy()[0:100],'x0_samp':x0_samp.detach().cpu().numpy()[0:20],
                              'x0_hat':x0_hat.detach().cpu().numpy()[0:20],'z0_samp':z0_samp.detach().cpu().numpy()[0:100],'z1_samp':z1_samp.detach().cpu().numpy()[0:100],
                              'coeff_spread0':coeff_spread0.detach().cpu().numpy()[0:100],'coeff_spread1':coeff_spread1.detach().cpu().numpy()[0:100],'total_kl_loss':total_kl_loss[:counter]})

    print('Before net save: ' + str(time.time()-pre_time))
    if (np.mod(j+1,100) == 0 or j+1 == total_epochs):
        saveName = save_dir + 'run{}_modelDict_{}_M{}Z{}_step{}.pt'.format(run_number, dataset, dict_size, latent_dim,counter)
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'zeta_decoder':zeta_decoder.state_dict(),
            'transOp': transOp.state_dict(),
            'classifier':classifier.state_dict(),
            'supervision': supervision,
            'total_con0_loss':total_con0_loss,
            'total_con1_loss':total_con1_loss,
            'latent_scale': latent_scale,
            'zeta': zeta,
            'gamma': gamma,
            'train_samples': train_samples,
        }, saveName)
    print('Before img save: ' + str(time.time()-pre_time))

compute_cSpread(encoder,decoder,train_loader,zeta_decoder,test_dir,device,counter,batch_size,args)
