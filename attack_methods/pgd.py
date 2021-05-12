"""
PGD
in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
import numpy as np
from attacks import Attack

class PGD(Attack):
    def __init__(self, target_cls, args):
        super(PGD, self).__init__("PGD", target_cls)
        self.eps = args.eps
        # for adversarial training
        if args.adv_train:
            self.eps = np.random.uniform(0.2, 0.5) if 'mnist' in args.dataset else np.random.uniform(0.02, 0.05)
        self.alpha = self.eps/4 # 2/255
        self.n_iters = args.pgd_iters
        self.random_start = args.pgd_random_start
        self.criterion = nn.CrossEntropyLoss()
        self.device = args.device

    def forward(self, imgs, labels, norm_fn, m, s):
        adv_imgs = imgs.clone().detach()
        adv_imgs.requires_grad = True

        if self.random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        for _ in range(self.n_iters):
            outputs, _ = self.target_cls(norm_fn(adv_imgs, m, s))
            loss = self.criterion(outputs, labels)

            grad = torch.autograd.grad(loss, adv_imgs)[0]

            adv_imgs = adv_imgs + self.alpha*grad.sign()
            eta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + eta, 0, 1)

        return adv_imgs.detach(), labels
