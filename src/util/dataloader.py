import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# Dataset that returns images with nearest neighbor
class NaturalTransformationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        super(NaturalTransformationDataset, self)
        self.dataset = dataset
        self.nn_graph = torch.arange(len(self.dataset))[:, None]

    def set_nn_graph(self, nn_graph):
        self.nn_graph = nn_graph

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x0, label = self.dataset.__getitem__(idx)
        neighbor = random.randrange(len(self.nn_graph[idx]))
        x1 = self.dataset.__getitem__(int(self.nn_graph[idx, neighbor]))
        return (x0, x1[0], label)

# Dataset that returns images with nearest neighbor
class IndexDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        super(IndexDataset, self)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, label = self.dataset.__getitem__(idx)
        return (x, label, idx)

def load_index_dataset(path, batch_size, img_indices, dataset="mnist", shuffle=True, data_type = 'test'):
    if dataset == "mnist":
        train_data = torchvision.datasets.MNIST(path, download=False, train=True,
                                                 transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(path, download=False, train=False,
                                                 transform=torchvision.transforms.ToTensor())
    elif dataset == "fmnist":
        train_data = torchvision.datasets.FashionMNIST(path, download=True, train=True,
                                                       transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.FashionMNIST(path, download=True,  train=False,
                                                      transform=torchvision.transforms.ToTensor())
    elif dataset == "svhn":
        train_data = torchvision.datasets.SVHN(path, download=True, split = 'train',
                                               transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.SVHN(path, download=True,  split = 'test',
                                              transform=torchvision.transforms.ToTensor())
    elif dataset == "celeba":
        train_data = torchvision.datasets.CelebA(path, split='train', target_type = 'attr',
                                          download=False,
                                          transform=transforms.Compose([
                        
                        transforms.CenterCrop(128),
                                          transforms.Resize(64),
                                          transforms.ToTensor(),
                                          ]))
        test_data = torchvision.datasets.CelebA(path, split='test', target_type = 'attr',
                                          download=False,
                                          transform=transforms.Compose([
                                          transforms.CenterCrop(128),
                                          transforms.Resize(64),
                                          transforms.ToTensor(),
                                          ]))
    if data_type == 'train':
        train_data = IndexDataset(train_data)
        train_data = torch.utils.data.Subset(train_data, img_indices)
        data_loader_use = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=2)
    elif data_type == 'test': 
        test_data = IndexDataset(test_data)
        test_data = torch.utils.data.Subset(test_data, img_indices)
        data_loader_use = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    return data_loader_use

def load_mnist(path,batch_size, train_images, test_images, train_classes=None, shuffle = True):

    mnist_train = torchvision.datasets.MNIST(path, download=True, train=True,
                                             transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST(path, download=False, train=False,
                                             transform=torchvision.transforms.ToTensor())
    mnist_train = NaturalTransformationDataset(mnist_train)
    mnist_test = NaturalTransformationDataset(mnist_test)

    if train_classes is not None:
        train_idx = np.in1d(mnist_train.dataset.targets, train_classes)
        mnist_train.dataset.data = mnist_train.dataset.data[train_idx]
        mnist_train.dataset.targets = np.array(mnist_train.dataset.targets)[train_idx]

        test_idx = np.in1d(mnist_test.dataset.targets, train_classes)
        mnist_test.dataset.data = mnist_test.dataset.data[test_idx]
        mnist_test.dataset.targets = np.array(mnist_test.dataset.targets)[test_idx]
        new_targets = np.arange(len(train_classes))

        # Map targets back down to [0, N]
        for i, t in enumerate(train_classes):
            mnist_train.dataset.targets[mnist_train.dataset.targets == t] = new_targets[i]
            mnist_test.dataset.targets[mnist_test.dataset.targets == t] = new_targets[i]

    mnist_train_subset = torch.utils.data.Subset(mnist_train, torch.arange(train_images))
    mnist_test_subset = torch.utils.data.Subset(mnist_test, torch.arange(test_images))

    train_loader = torch.utils.data.DataLoader(mnist_train_subset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=2)
    test_loader = torch.utils.data.DataLoader(mnist_test_subset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=2)

    return (train_loader, test_loader)

def load_svhn(path,batch_size, train_samples=None, train_classes=None,shuffle=True):
    transform = transforms.ToTensor()
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614))])
    svhn_train = torchvision.datasets.SVHN(path, download=True, split = 'train',
                                             transform=transform)
    svhn_test = torchvision.datasets.SVHN(path, download=True,  split = 'test',
                                             transform=transform)
    trainset= NaturalTransformationDataset(svhn_train)
    testset  = NaturalTransformationDataset(svhn_test)

    if train_classes is not None:
        train_idx = np.in1d(trainset.dataset.labels, train_classes)
        trainset.dataset.data = trainset.dataset.data[train_idx]
        trainset.dataset.labels = np.array(trainset.dataset.labels)[train_idx]

        test_idx = np.in1d(testset.dataset.labels, train_classes)
        testset.dataset.data = testset.dataset.data[test_idx]
        testset.dataset.labels = np.array(testset.dataset.labels)[test_idx]
        new_targets = np.arange(len(train_classes))

        # Map targets back down to [0, N]
        for i, t in enumerate(train_classes):
            trainset.dataset.labels[trainset.dataset.labels == t] = new_targets[i]
            testset.dataset.labels[testset.dataset.labels == t] = new_targets[i]

    if train_samples is not None:
        trainset = torch.utils.data.Subset(trainset, torch.arange(0, int(train_samples)))
    else:
        trainset = torch.utils.data.Subset(trainset, torch.arange(0, len(svhn_train.data)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return (train_loader, test_loader)

def load_fmnist(path,batch_size, train_samples=None, train_classes=None,shuffle=True):
    transform = transforms.ToTensor()
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5,), (0.5,))])
    fmnist_train = torchvision.datasets.FashionMNIST(path, download=True, train=True,
                                             transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(path, download=True,  train=False,
                                             transform=transform)
    trainset= NaturalTransformationDataset(fmnist_train)
    testset  = NaturalTransformationDataset(fmnist_test)

    if train_classes is not None:
        train_idx = np.in1d(trainset.dataset.targets, train_classes)
        trainset.dataset.data = trainset.dataset.data[train_idx]
        trainset.dataset.targets = np.array(trainset.dataset.targets)[train_idx]

        test_idx = np.in1d(testset.dataset.targets, train_classes)
        testset.dataset.data = testset.dataset.data[test_idx]
        testset.dataset.targets = np.array(testset.dataset.targets)[test_idx]
        new_targets = np.arange(len(train_classes))

        # Map targets back down to [0, N]
        for i, t in enumerate(train_classes):
            trainset.dataset.targets[trainset.dataset.targets == t] = new_targets[i]
            testset.dataset.targets[testset.dataset.targets == t] = new_targets[i]

    if train_samples is not None:
        trainset = torch.utils.data.Subset(trainset, torch.arange(0, int(train_samples)))
    else:
        trainset = torch.utils.data.Subset(trainset, torch.arange(0, len(fmnist_train.data)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    return (train_loader, test_loader)

def load_cifar10(path, batch_size, train_samples=None, train_classes=None,shuffle=True):
    transform = transforms.ToTensor()
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar_train = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=False, transform=transform)

    if train_classes is not None:
        train_idx = np.in1d(cifar_train.targets, train_classes)
        cifar_train.data = cifar_train.data[train_idx]
        cifar_train.targets = np.array(cifar_train.targets)[train_idx]

        test_idx = np.in1d(cifar_test.targets, train_classes)
        cifar_test.data = cifar_test.data[test_idx]
        cifar_test.targets = np.array(cifar_test.targets)[test_idx]
        new_targets = np.arange(len(train_classes))

        # Map targets back down to [0, N]
        for i, t in enumerate(train_classes):
            cifar_train.targets[cifar_train.targets == t] = new_targets[i]
            cifar_test.targets[cifar_test.targets == t] = new_targets[i]

    trainset = NaturalTransformationDataset(cifar_train)
    testset = NaturalTransformationDataset(cifar_test)

    if train_samples is not None:
        trainset = torch.utils.data.Subset(trainset, torch.arange(0, int(train_samples)))
        testset = torch.utils.data.Subset(testset, torch.arange(0, int(train_samples)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return (train_loader, test_loader)

def load_celeba(path, batch_size, train_samples=None, train_classes=None):
    train_data = torchvision.datasets.CelebA('./data', split='train', target_type = 'attr',
                                  download=True,
                                  transform=transforms.Compose([
                                  transforms.CenterCrop(128),
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  ]))
    test_data = torchvision.datasets.CelebA('./data', split='test', target_type = 'attr',
                                  download=True,
                                  transform=transforms.Compose([
                                  transforms.CenterCrop(128),
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  ]))

    train_data = NaturalTransformationDataset(train_data)
    test_data = NaturalTransformationDataset(test_data)
    if len(train_data) > train_samples:
        train_data = torch.utils.data.Subset(train_data, torch.arange(0, int(train_samples)))
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)
    return (train_loader, test_loader)

def load_celeba64(path, batch_size, train_samples=None, train_classes=None):
    train_data = torchvision.datasets.CelebA(path, split='train', target_type = 'attr',
                                  download=False,
                                  transform=transforms.Compose([
                                  transforms.CenterCrop(128),
                                  transforms.Resize(64),
                                  transforms.ToTensor(),
                                  ]))
    test_data = torchvision.datasets.CelebA(path, split='test', target_type = 'attr',
                                  download=False,
                                  transform=transforms.Compose([
                                  transforms.CenterCrop(128),
                                  transforms.Resize(64),
                                  transforms.ToTensor(),
                                  ]))

    train_data = NaturalTransformationDataset(train_data)
    test_data = NaturalTransformationDataset(test_data)
    if len(train_data) > train_samples:
        train_data = torch.utils.data.Subset(train_data, torch.arange(0, int(train_samples)))
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    return (train_loader, test_loader)
