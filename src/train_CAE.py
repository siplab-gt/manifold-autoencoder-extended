import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

from model.transop import TransOp_expm
from model.l1_inference import infer_coefficients
from model.autoencoder import ConvEncoder, ConvDecoder, init_weights
from util.dataloader import load_mnist, load_cifar10, load_celeba, load_celeba64, load_fmnist

parser = argparse.ArgumentParser()
parser.add_argument('-Z', '--latent_dim', default=10, type=int, help="Dimension of latent space")
parser.add_argument('-d', '--dataset', default='mnist', type=str, help="Dataset to Use ['cifar10', 'mnist','fmnist','svhn']")
parser.add_argument('-N', '--train_samples', default=50000, type=int, help="Number of training samples to use.")
parser.add_argument('-L', '--Lambda', default=1e-4, type=float, help="Contractive penalty weighting.")
args = parser.parse_args()

dataset = args.dataset
latent_dim = args.latent_dim
Lambda = args.Lambda
batch_size = 500
ae_epochs = 200
train_samples = args.train_samples
train_classes = np.arange(10)

# Create folder for logging if it does not exist
save_dir = f'../results/CAE_{dataset}_{latent_dim}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dataset == 'mnist':
    test_imgs = 10000
    train_loader, test_loader = load_mnist('./data',batch_size, train_samples, test_imgs)
    channels, image_dim, features = 1, 28, 64
    num_classes = len(train_classes)
elif dataset == 'cifar10_vehicle':
    train_loader, test_loader = load_cifar10('./data', batch_size, train_samples=train_samples, train_classes=[0, 1, 8, 9])
    channels, image_dim, features = 3, 32, 128
    num_classes = 4
elif dataset == 'cifar10_animal':
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
    class_epochs = 0
    channels, image_dim, features = 3, 32, 128
    num_classes = len(train_classes)
elif dataset == 'celeba64':
    train_loader, test_loader = load_celeba64('./data', batch_size, train_samples=train_samples, train_classes=train_classes)
    channels, image_dim, features = 3, 64, 128
    num_classes = len(train_classes)

encoder = ConvEncoder(latent_dim, channels, image_dim, False, num_filters=features).to(device)
decoder = ConvDecoder(latent_dim, channels, image_dim, num_filters=features).to(device)
encoder.apply(init_weights)
decoder.apply(init_weights)
autoenc_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=1e-4, betas=(0.5, 0.999))
ae_scheduler = torch.optim.lr_scheduler.MultiStepLR(autoenc_opt, milestones=[15, 50, 75, 100, 150], gamma=0.2)

for j in range(ae_epochs):
    encoder.train()
    decoder.train()
    for idx, batch in enumerate(train_loader):
        x0, _, __ = batch
        x0 = x0.to(device)

        # Require gradient for image for computing Jacobian
        x0.requires_grad_(True)
        x0.retain_grad()

        z0 = encoder(x0)
        x0_hat = decoder(z0)

        # Recon loss
        ae_loss = F.mse_loss(x0_hat, x0, reduction='mean')

        # jacobian wrt the input
        z0.backward(torch.ones(z0.size()).to(device), retain_graph=True)
        jacobian = torch.sqrt(torch.sum(torch.pow(x0.grad, 2)))
        x0.grad.data.zero_()
        x0.requires_grad_(False)

        # Total loss, the sum of the two loss terms, with weight applied to second term
        autoenc_opt.zero_grad()
        total_loss = ae_loss + (Lambda*jacobian)
        total_loss.backward()
        autoenc_opt.step()

    encoder.eval()
    decoder.eval()
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
    ae_scheduler.step()

torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
}, save_dir + 'CAE_{}_Z{}_Lambda{}.pt'.format(dataset, latent_dim, Lambda))

num_samples = 10
num_directions = 5
coeff_range = 3

x, _, __ = next(iter(train_loader))
x = x[:num_samples]
x = x.to(device)

def find_jacobian(encoder, x):
    # Require gradient for image for computing Jacobian
    x.requires_grad_(True)
    x.retain_grad()
    z = encoder(x)

    jacobian = torch.zeros((*x.shape, z.shape[1]))
    for d in range(latent_dim):
        x_hat = decoder(z)
        one_code = torch.zeros(z.size())
        one_code[:, d] = 1
        z.backward(one_code.to(device), retain_graph=True)
        jacobian[:, :, :, :, d] = x.grad.detach()
        x.grad.data.zero_()
    return jacobian

jacobian = find_jacobian(encoder, x)

for n in range(num_samples):
    u, s, v = torch.svd(jacobian[n].reshape(-1, latent_dim))
    z = encoder(x)[n]
    fig, ax = plt.subplots(nrows=5, ncols=11, figsize=(25, 15))

    for d in range(num_directions):
        ax[d, 5].imshow(x[n].permute(1, 2, 0).detach().cpu().squeeze().numpy())
        direction = v.T[:, d].to(device)

        coeff = torch.linspace(0, coeff_range, 5).float().to(device)
        x_hat = decoder(z[None, :] + coeff[:, None]*direction).detach().cpu()
        for k in range(len(x_hat)):
            ax[d, 6+k].imshow(x_hat[k].permute(1, 2, 0).squeeze().numpy())

        coeff = torch.linspace(-coeff_range, 0, 5).float().to(device)
        x_hat = decoder(z[None, :] + coeff[:, None]*direction).detach().cpu()
        for k in range(len(x_hat)):
            ax[d, k].imshow(x_hat[k].permute(1, 2, 0).squeeze().numpy())

    [axi.set_axis_off() for axi in ax.ravel()]
    plt.savefig(save_dir + f'sample_augmentations{n}_lam{Lambda}.png', bbox_inches='tight')
    plt.close()
