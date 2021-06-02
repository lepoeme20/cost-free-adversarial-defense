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
    def __init__(self, target_cls, args, train_loader, blackbox_cls=None):
        super(PGD, self).__init__("PGD", target_cls, blackbox_cls)
        self.eps = args.eps
        self.alpha = self.eps/4
        self.n_iters = args.pgd_iters
        self.random_start = args.pgd_random_start
        self.criterion_CE = nn.CrossEntropyLoss()
        self.blackbox_cls = blackbox_cls
        self.adaptive = args.adaptive
        if self.adaptive:
            m, s = get_m_s(args)
            center = get_center(
                target_cls, train_loader, args.num_class, args.device, m, s
            )
            self.criterion = Loss(
                args.num_class,
                args.device,
                pre_center=center,
                phase='intra',
            )

    def forward(self, imgs, labels, norm_fn, m, s):
        adv_imgs = imgs.clone().detach()
        adv_imgs.requires_grad = True

        if self.random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        for _ in range(self.n_iters):
            if self.blackbox_cls is not None:
                outputs, features = self.blackbox_cls(norm_fn(adv_imgs, m, s))
            else:
                outputs, features = self.target_cls(norm_fn(adv_imgs, m, s))

            if self.adaptive:
                ce_loss = self.criterion_CE(outputs, labels)
                intra_loss = self.criterion(features, labels)
                loss = ce_loss + intra_loss
            else:
                loss = self.criterion_CE(outputs, labels)

            grad = torch.autograd.grad(loss, adv_imgs)[0]

            adv_imgs = adv_imgs + self.alpha*grad.sign()
            eta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + eta, 0, 1)

        return adv_imgs.detach(), labels
