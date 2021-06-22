import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def soft_threshold(c, zeta):
    return F.relu(torch.abs(c) - zeta) * torch.sign(c)

def compute_loss(c, x0, x1, psi):
    T = (psi[None, :, :, :] * c[:, :, None, None]).sum(dim=1).reshape((
        x0.shape[0], psi.shape[1], psi.shape[2]))
    x1_hat = torch.matrix_exp(T) @ x0
    loss = F.mse_loss(x1_hat, x1, reduction='sum')
    return loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def infer_coefficients(x0, x1, psi, zeta, max_iter=800, tol=1e-5, device='cpu'):
    c = nn.Parameter(torch.mul(torch.randn((len(x0),len(psi)), device=device),
                     0.02), requires_grad=True)
    c_opt = torch.optim.SGD([c], lr=1e-2, nesterov=True, momentum=0.9)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(c_opt, gamma=0.985)
    change = 1e99
    k = 0
    while k < max_iter and change > tol:
        old_coeff = c.clone()

        c_opt.zero_grad()
        loss = compute_loss(c, x0, x1, psi)
        loss.backward()
        c_opt.step()
        opt_scheduler.step()

        with torch.no_grad():
            c.data = soft_threshold(c, get_lr(c_opt)*zeta)

        change = torch.norm(c.data - old_coeff) / (torch.norm(old_coeff) + 1e-9)
        k += 1
    return (loss.item(), get_lr(c_opt), k), c.data

# V1 Line search
# source: http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf
def infer_coefficients_(x0, x1, psi, zeta, max_iter=600, tol=1e-4, consistency=None, device='cpu'):
    c = .1*torch.randn(x0.shape[0], psi.shape[0], dtype=psi.dtype).to(device)

    k = 1
    beta = 0.85
    t = 1e-2
    old_c = c.clone()
    change = 1e99

    # Check
    while k < max_iter and change > tol:
        # compute momentum term
        if k == 1:
            v = old_c.clone().detach().requires_grad_(True)
        else:
            v = (c + ((k-2)/(k+1))*(c - old_c)).clone().detach().requires_grad_(True)

        # Save previous iterate of c
        old_c = c.clone()
        # Compute current loss and automatically differentiate gradient
        loss = compute_loss(v, x0, x1, psi, consistency)
        loss.backward()

        # Take forward and backward gradient steps
        c_new = soft_threshold(c - t*v.grad, t*zeta)
        # Compute new loss and perform backsearch to tune step_size
        c_loss = compute_loss(c_new, x0, x1, psi, consistency)
        Q = (v.grad[:, None, :] @ (c_new - v)[:, :, None]).sum()
        while c_loss > loss + Q  + (1/(2*t))*(torch.norm(c_new - v)**2):
            t = beta*t
            c_new = soft_threshold(c - t*v.grad, t*zeta)
            c_loss = compute_loss(c_new, x0, x1, psi, consistency)
            Q = (v.grad[:, None, :] @ (c_new - v)[:, :, None]).sum()

        # accept new c value
        c = c_new

        # Compute change to detect exit condition
        change = torch.norm(c - old_c) / torch.norm(old_c)
        k += 1

    return c_loss, t, k, c.data

def compute_arc_length(psi,c,t,x0,device = 'cpu'):
    batch_size = x0.shape[0]
    N = x0.shape[1]
    A = (psi[None,:,:,:] * c[:,:,None,None]).sum(dim=1).reshape((batch_size,N,N))
    arc_len = torch.zeros((batch_size)).to(device)
    #x0 = torch.transpose(x0,0,1)
    t_int = t[1]-t[0]
    count = 0
    for t_use in t:
        #Tx = torch.matrix_exp(A*t_use) @ x0
        Tx = torch.matmul(torch.matrix_exp(A*t_use), x0.unsqueeze(-1))
        A_Tx = torch.matmul(A,Tx).squeeze()
        arc_len = arc_len + t_int*torch.norm(A_Tx,dim = 1)


    return arc_len.detach().cpu().numpy()
