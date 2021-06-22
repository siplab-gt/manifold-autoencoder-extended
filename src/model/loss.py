# Computation of loss functions used from here: https://github.com/GT-RIPL/FeatMatch.git
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-20

def log_loss(logits_p, prob_p, logits_q, prob_q):
    eps = 1e-12
    prob_p = prob_p if prob_p is not None else F.softmax(logits_p, dim=1)
    logq = F.log_softmax(logits_q, dim=1) if logits_q is not None else torch.log(prob_q + eps)
    return -torch.mean(torch.sum(prob_p.detach() * logq, dim=1))

def l2_loss(logits_p, prob_p, logits_q, prob_q):
    prob_p = prob_p if prob_p is not None else F.softmax(logits_p, dim=1)
    prob_q = prob_q if prob_q is not None else F.softmax(logits_q, dim=1)
    L = prob_p.size(1)
    return torch.mean(torch.sum((prob_p.detach() - prob_q)**2, dim=1))/L


def kld_loss(logits_p, prob_p, logits_q, prob_q):
    prob_p = prob_p if prob_p is not None else F.softmax(logits_p, dim=1)
    logp = F.log_softmax(logits_p, dim=1) if logits_p is not None else torch.log(prob_p+eps)
    logq = F.log_softmax(logits_q, dim=1) if logits_q is not None else torch.log(prob_q+eps)

    return torch.mean(torch.sum(prob_p * (logp - logq), dim=1))

def kld_loss_mod(logits_p, prob_p, logits_q, prob_q):
    prob_p = prob_p if prob_p is not None else F.softmax(logits_p, dim=1)
    logp = F.log_softmax(logits_p, dim=1) if logits_p is not None else torch.log(prob_p+eps)
    logq = F.log_softmax(logits_q, dim=1) if logits_q is not None else torch.log(prob_q+eps)

    return torch.mean(torch.sum(prob_p * (logp - logq)**2, dim=1))


def jsd_loss(logits_p, prob_p, logits_q, prob_q):
    prob_p = prob_p if prob_p is not None else F.softmax(logits_p, dim=1)
    prob_q = prob_q if prob_q is not None else F.softmax(logits_q, dim=1)
    prob_m = (prob_p + prob_q)/2.

    return (kld_loss(logits_p, prob_p, None, prob_m) + kld_loss(logits_q, prob_q, None, prob_m))/2.