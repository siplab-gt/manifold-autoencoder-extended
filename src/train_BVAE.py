import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

from model.transop import TransOp_expm
from model.autoencoder import BetaVAE
from util.dataloader import load_mnist, load_cifar10, load_celeba, load_celeba64, load_svhn, load_fmnist

parser = argparse.ArgumentParser()
parser.add_argument('-Z', '--latent_dim', default=32, type=int, help="Dimension of latent space")
parser.add_argument('-d', '--dataset', default='celeba64', type=str, help="Dataset to Use ['cifar10', 'mnist','fmnist','svhn']")
parser.add_argument('-N', '--train_samples', default=150000, type=int, help="Number of training samples to use.")

args = parser.parse_args()
dataset = args.dataset
latent_dim = args.latent_dim
batch_size = 100
ae_epochs = 200
train_samples = args.train_samples
train_classes = np.arange(10)

# Create folder for logging if it does not exist
save_dir = f'./results/BVAE_{dataset}_{latent_dim}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data
if dataset == 'mnist':
    test_imgs = 10000
    train_loader, test_loader = load_mnist('./data',batch_size, train_samples, test_imgs)
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
    beta0 = [0.01, 0.1, 0.5,1, 1,1, 5, 5,10]
elif dataset == 'cifar10_vehicle':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[0, 1, 8, 9])
    channels, image_dim, features = 3, 32, 128
    num_classes = 4
    beta0 = [0.1, 1, 10, 100, 200, 250]
elif dataset == 'cifar10_animal':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[3, 4, 5, 7])
    channels, image_dim, features = 3, 32, 128
    num_classes = 4
    beta0 = [0.1, 1, 10, 100, 200, 250]
elif dataset == 'cifar10':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    channels, image_dim, features = 3, 32, 128
    num_classes = len(train_classes)
    beta0 = [0.1, 1, 10, 100, 200, 250]
elif dataset == 'svhn':
    train_loader, test_loader = load_svhn('./data', batch_size, train_samples =train_samples,train_classes=train_classes)
    channels, image_dim, features = 3, 32, 128
    num_classes = len(train_classes)
    beta0 = [0.1, 1, 10, 100, 200, 250]
elif dataset == 'fmnist':
    train_loader, test_loader = load_fmnist('./data', batch_size, train_classes=train_classes)
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
    beta0 = [0.01, 0.1, 0.5,1, 1,1, 5, 5,10]
    #beta0 = [0.01, 0.1, 0.5, 1, 5, 5]
elif dataset == 'celeba':
    train_loader, test_loader = load_celeba('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    class_epochs = 0
    channels, image_dim, features = 3, 32, 128
    num_classes = len(train_classes)
    beta0 = [0.1, 1, 10, 100, 200, 250]
elif dataset == 'celeba64':
    train_loader, test_loader = load_celeba64('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    class_epochs = 0
    channels, image_dim, features = 3, 64, 128
    num_classes = len(train_classes)
    #beta0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    beta0 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

beta = beta0[0]
bvae = BetaVAE(z_dim=latent_dim, nc=channels, img_size = image_dim).to(device)
autoenc_opt = torch.optim.Adam(bvae.parameters(),
                               lr=1e-4, betas=(0.9, 0.999))
ae_scheduler = torch.optim.lr_scheduler.MultiStepLR(autoenc_opt, milestones=[15, 50, 75, 100, 150], gamma=0.2)
beta = 0

for j in range(ae_epochs):
    if j % 25 == 0:
        print(f"UPGRADE BETA AT {j}")
        torch.save({
            'bvae': bvae.state_dict(),
        }, save_dir + 'BVAE_{}_Z{}_Beta{}.pt'.format(dataset, latent_dim, beta))

        beta = beta0[j // 25]

    bvae.train()
    for idx, batch in enumerate(train_loader):
        x0, _, __ = batch
        x0 = x0.to(device)

        x0_hat, z_mean, z_scale = bvae(x0)

        # Recon loss
        ae_loss = F.mse_loss(x0_hat, x0, reduction='sum')
        # KL loss
        kl_loss = -0.5 * torch.sum(1 + z_scale - (z_mean ** 2) - z_scale.exp())

        # Total loss, the sum of the two loss terms, with weight applied to second term
        autoenc_opt.zero_grad()
        total_loss = ae_loss + (beta*kl_loss)
        total_loss.backward()
        autoenc_opt.step()

    bvae.eval()
    test_error = torch.zeros(len(test_loader))
    kl_error = torch.zeros(len(test_loader))

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x0, _, __ = batch
            x0 = x0.to(device)
            x0_hat, z_mean, z_scale = bvae(x0)

            # Recon loss
            ae_loss = F.mse_loss(x0_hat, x0, reduction='sum')
            # KL loss
            kl_loss = -0.5 * torch.sum(1 + z_scale - (z_mean ** 2) - z_scale.exp())

            test_error[idx] = ae_loss
            kl_error[idx] = kl_loss
    print("AE Pre-train epoch {} of {}: Test loss: {:.4E} KL loss: {:.4E} beta: {:.2E}".format(j+1, ae_epochs,
                                                                                  torch.mean(test_error),
                                                                                  torch.mean(kl_error),
                                                                                  beta))
    ae_scheduler.step()


    if j % 25 == 0:
        x, _, __ = next(iter(train_loader))
        x = x.to(device)
        x_hat1, z_mean, z_scale = bvae(x)
        z = z_mean + torch.randn_like(z_scale)*torch.exp(0.5*z_scale)

        fig, ax = plt.subplots(nrows=32, ncols=10, figsize=(15, 35))
        sweep_range = 12
        interpolation = torch.arange(-sweep_range, sweep_range, sweep_range/5)

        for j in range(latent_dim):
            for i, val in enumerate(interpolation):
                z_clone = z.clone()
                z_clone[:, j] = val
                im = bvae._decode(z_clone).detach().cpu()
                ax[j, i].imshow(im[0].permute(1, 2, 0).detach().cpu().numpy())
        [axi.set_axis_off() for axi in ax.ravel()]
