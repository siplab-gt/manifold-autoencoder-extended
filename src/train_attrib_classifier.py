import re
import os
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from util.dataloader import load_cifar10
from model.attribute_resnet import resnet18, resnet50

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--train_samples', default=None, type=str, help="Number of training samples to use")

# PARSE ARGUMENTS #
args = parser.parse_args()
epochs = 100
network_lr = 1e-3
batch_size = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create folder for logging if it does not exist
save_dir = time.strftime(f"./results/CelebA_AttribClf_%m.%d.%y/")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for figures at {}".format(save_dir))

# DATA LOADER #

train_data = torchvision.datasets.CelebA('./data', split='train', target_type = 'attr',
                              download=True,
                              transform=transforms.Compose([
                              transforms.CenterCrop(128),
                              transforms.Resize(64),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              ]))
val_data = torchvision.datasets.CelebA('./data', split='valid', target_type = 'attr',
                              download=True,
                              transform=transforms.Compose([
                              transforms.CenterCrop(128),
                              transforms.Resize(64),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              ]))
test_data = torchvision.datasets.CelebA('./data', split='test', target_type = 'attr',
                              download=True,
                              transform=transforms.Compose([
                              transforms.CenterCrop(128),
                              transforms.Resize(64),
                              transforms.ToTensor(),
                              ]))

if args.train_samples is not None:
    train_data = torch.utils.data.Subset(train_data, torch.arange(0, int(args.train_samples)))

train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)
val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)
img_size = (32, 32)
channels = 3

print(f"Traindata size: {len(train_loader.dataset)} test data size: {len(test_loader.dataset)}")

# INITIALIZE MODELS #
model = resnet50().to(device)
opt = torch.optim.SGD(model.parameters(), lr=network_lr, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
criterion = nn.CrossEntropyLoss()

# INITIALIZE LOGGING VARIABLES
objective_indices = [4, 5, 8, 9, 11, 15, 16, 17, 20, 21, 22, 24, 26, 28, 30, 31]
total_loss =  np.zeros(epochs)
total_acc =  np.zeros((epochs, len(objective_indices)))
total_time = np.zeros(epochs)
val_loss = np.zeros(len(objective_indices))
weighting = np.ones(len(objective_indices))

# BEGIN TRAINING #
for j in range(epochs):
    epoch_time = time.time()
    model.train()
    for i, batch in enumerate(train_loader):
        # Move data memory and encode latent point and draw from prior
        x,  label = batch
        x, label = x.to(device), label.to(device)
        out = model(x)

        loss = []
        for k, a in enumerate(objective_indices):
            loss.append(weighting[k] * criterion(out[a], label[:, a]))

        loss_tot = sum(loss)
        opt.zero_grad()
        loss_tot.backward()
        opt.step()

    total_time[j] = time.time() - epoch_time

    # Test reconstruction and discriminator loss on test dataset
    model.eval()
    with torch.no_grad():

        epoch_val_loss = np.zeros((len(val_loader), len(objective_indices)))
        for i, batch in enumerate(val_loader):
            x, label = batch
            x, label = x.to(device), label.to(device)
            out = model(x)
            for k, a in enumerate(objective_indices):
                epoch_val_loss[i, k] = ((out[a] - label[:, a, None])**2).sum(dim=-1).mean()
        if j > 0:
            weighting = np.abs((np.mean(epoch_val_loss, axis=0) - val_loss) / val_loss)
        val_loss = np.mean(epoch_val_loss, axis=0)

        epoch_loss = np.zeros(len(test_loader))
        epoch_correct = np.zeros((len(test_loader), len(objective_indices)))
        for i, batch in enumerate(test_loader):
            x, label = batch
            x, label = x.to(device), label.to(device)
            out = model(x)

            loss = []
            correct = torch.zeros(len(objective_indices))
            for k, a in enumerate(objective_indices):
                loss.append(criterion(out[a], label[:, a]))
                _, pred = out[a].topk(1, 1, True, True)
                correct[k] = pred[0].eq(label[:, a]).sum()

            epoch_loss[i] = sum(loss)
            epoch_correct[i] = correct

    total_loss[j] = np.mean(epoch_loss)
    total_acc[j] = 100 * (epoch_correct.sum(axis=0) / len(test_loader.dataset))
    best_acc, best_class = total_acc[j].max(), objective_indices[total_acc[j].argmax()]
    worst_acc, worst_class = total_acc[j].min(), objective_indices[total_acc[j].argmin()]
    print(f"Epoch {j+1} of {epochs}, time: {total_time[j]:.2f}, val loss: {total_loss[j]:.2E}, mean acc: {total_acc[j].mean():.2f}%, best attr: {best_class} at {best_acc:.2f}%, worst attr: {worst_class} at {worst_acc:.2f}%")

    # Tune step size for autoencoder and discriminator
    scheduler.step()

    # Save models and logging information
    torch.save({
        'classifier': model.state_dict(),
        'total_loss': total_loss,
        'total_time': total_time,
    }, save_dir + f'Celeba_AttrClf.pt')
