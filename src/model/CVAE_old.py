"""
    Simple convolutional VAE, suitable for MNIST experiments
"""

import numpy as np
import torch
import torch.nn as nn

def VAE_LL_loss(Xbatch,Xest,logvar,mu):
    batch_size = Xbatch.shape[0]
    sse_loss = torch.nn.MSELoss(reduction = 'sum') # sum of squared errors
    KLD = 1./batch_size * -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = 1./batch_size * sse_loss(Xest,Xbatch)
    auto_loss = mse + KLD
    return auto_loss, mse, KLD

"""
joint_uncond:
    Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
Inputs:
    - params['Nalpha'] monte-carlo samples per causal factor
    - params['Nbeta']  monte-carlo samples per noncausal factor
    - params['K']      number of causal factors
    - params['L']      number of noncausal factors
    - params['M']      number of classes (dimensionality of classifier output)
    - decoder
    - classifier
    - device
Outputs:
    - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
    - info['xhat']
    - info['yhat']
"""
def joint_uncond(params, decoder, classifier, device):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M'])
    zs = np.zeros((params['Nalpha']*params['Nbeta'], params['z_dim']))
    for i in range(0, params['Nalpha']):
        alpha = np.random.randn(params['K'])
        zs = np.zeros((params['Nbeta'],params['z_dim']))  
        for j in range(0, params['Nbeta']):
            beta = np.random.randn(params['L'])
            zs[j,:params['K']] = alpha
            zs[j,params['K']:] = beta
        # decode and classify batch of Nbeta samples with same alpha
        xhat = decoder(torch.from_numpy(zs).float().to(device))
        yhat = classifier(xhat)[0].detach().cpu()
        p = 1./float(params['Nbeta']) * torch.sum(yhat,0) # estimate of p(y|alpha)
        I = I + 1./float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p+eps)))
        q = q + 1./float(params['Nalpha']) * p # accumulate estimate of p(y)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    negCausalEffect = -I
    info = {"xhat" : xhat, "yhat" : yhat}
    return negCausalEffect, info

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Encoder(nn.Module):

    def __init__(self, z_dim, c_dim, x_dim,
                 filt_per_layer=64): # x_dim : total number of pixels
        super(Encoder, self).__init__()
     
        self.model_enc = nn.Sequential(
            nn.Conv2d(int(c_dim), filt_per_layer, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(filt_per_layer, filt_per_layer, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ZeroPad2d((1,2,1,2)),
            nn.Conv2d(filt_per_layer, filt_per_layer, 4, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(int(filt_per_layer*x_dim/16), z_dim)
        self.fc_var = nn.Linear(int(filt_per_layer*x_dim/16), z_dim)
    
    def encode(self, x):
        z = self.model_enc(x)
        z = z.view(z.shape[0], -1)
        return self.fc_mu(z), self.fc_var(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, z_dim, c_dim, x_dim, # x_dim : total number of pixels
                 filt_per_layer=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, int(filt_per_layer*self.x_dim/16)),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(filt_per_layer, filt_per_layer, 4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer, filt_per_layer, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer, int(c_dim), 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.shape[0]
        t = self.fc(z).view(batch_size, -1,
                            int(np.sqrt(self.x_dim)/4),
                            int(np.sqrt(self.x_dim)/4))
        x = self.model(t)
        return x
