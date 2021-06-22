
import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data



from util.nearestneighbors import neighbors
import numpy as np
import scipy.io as sio
import random


def find_nn(z_in, eval_loader, encoder, batch_size, device,ne=5):
    print('---- Finding NN ({}) ----'.format(ne))
    encoder.eval()
    cos_sim = nn.CosineSimilarity(dim = 1)
    nearest = neighbors(ne)

    for i, batch in enumerate(eval_loader):
        # Prepare the inputs
        input,_,target = batch
        with torch.no_grad():
            input = input.to(device)
            
            #target = target.cuda()
            #image = image.cuda()

        z_comp = encoder(input)
        sim = cos_sim(z_in,z_comp)
        
        # Inference
        batch_size_temp = input.shape[0]
        indices = torch.FloatTensor(range(i*batch_size,i*batch_size+batch_size_temp))
        # Update the performance meter

        nearest.update(input, sim.detach(), target,indices)


    #nearest.save('outputs/neighbors',image, tar, denormalize)
    return nearest

def get_neighbor_batch(input_data,distAll,labels,batch_size):
    N = input_data.shape[1]  
    numEx = input_data.shape[0]
    x0 = np.zeros((batch_size,N))
    x1 = np.zeros((batch_size,N))
    label_store = np.zeros((batch_size))
    for k in range(0,batch_size):
        x0Idx = random.randint(0,numEx-1)
        x0[k,:] = input_data[x0Idx,:]
        
        distPoss = distAll[x0Idx,:]
        sortIdx = np.argsort(distPoss)
        sampUse = random.randint(5,15); 
        idxUse = sortIdx[sampUse]
        x1[k,:] = input_data[idxUse,:]
        label_store[k] = labels[x0Idx]
              
        
    return x0,x1,label_store
