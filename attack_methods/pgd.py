"""
PGD
in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'

This code is written by Seugnwan Seo
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from attacks import Attack
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import Loss, get_center, get_m_s

class PGD(Attack):
    def __init__(self, target_cls, args):
        super(PGD, self).__init__("PGD", target_cls)
        self.eps = args.eps
        self.alpha = self.eps/4
        self.n_iters = args.pgd_iters
        self.random_start = args.pgd_random_start
        self.criterion_CE = nn.CrossEntropyLoss()
        self.model = target_cls

    def forward(self, imgs, labels, norm_fn, m, s):
        adv_imgs = imgs.clone().detach()

        if self.random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        for _ in range(self.n_iters):
            adv_imgs.requires_grad = True
            outputs, features = self.model(norm_fn(adv_imgs, m, s))

            loss = self.criterion_CE(outputs, labels)

            grad = torch.autograd.grad(
                loss, adv_imgs, retain_graph=False, create_graph=False
            )[0]

            adv_imgs = adv_imgs.detach() + self.alpha*grad.sign()
            eta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + eta, 0, 1).detach()

        return adv_imgs, labels
