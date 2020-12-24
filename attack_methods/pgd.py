"""[summary]

"""
import torch
import torch.nn as nn
import numpy as np
from attacks import Attack

class PGD(Attack):
    """Reproduce PGD
    in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'

    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        eps {float} -- For bound eta (difference between original imgs and adversarial imgs)
        alpha {float} -- Magnitude of perturbation (same as eps in FGSM)
        n_iters {int} -- Step size
        random_start {bool} -- If ture, initialize perturbation using eps
    """
    def __init__(self, target_cls, args):
        super(PGD, self).__init__("PGD", target_cls)
        self.eps = args.eps
        self.alpha = 2/255
        self.n_iters = args.pgd_iters
        self.random_start = args.pgd_random_start
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, imgs, labels, norm_fn, m, s):
        if self.args.adv_train:
            self.eps = np.random.uniform(0.02,0.05)

        imgs = imgs.to(self.args.device)
        labels = labels.to(self.args.device)

        adv_imgs = imgs.clone().detach()
        adv_imgs.requires_grad = True

        if self.random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        for _ in range(self.n_iters):
            outputs, _, _, _ = self.target_cls(norm_fn(adv_imgs, m, s))
            loss = self.criterion(outputs, labels)

            grad = torch.autograd.grad(loss, adv_imgs)[0]

            adv_imgs = adv_imgs + self.alpha*grad.sign()
            eta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + eta, 0, 1)

        return adv_imgs.detach(), labels
