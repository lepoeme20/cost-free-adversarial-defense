"""
Iterative FGSM

This code is written by Seugnwan Seo
"""
import os
import sys
import torch
import torch.nn as nn
from attacks import Attack
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import Loss, get_center, get_m_s


class MIM(Attack):
    def __init__(self, target_cls, args):
        super(MIM, self).__init__("MIM", target_cls)
        self.eps = args.eps
        self.alpha = args.eps/args.mim_step
        self.step = args.bim_step
        self.criterion_CE = nn.CrossEntropyLoss()
        self.model = target_cls

    def forward(self, imgs, labels, norm_fn, m, s):
        imgs = imgs.clone().detach()
        adv_imgs = imgs.clone().detach()

        momentum = torch.zeros_like(imgs).detach().to(imgs.device)

        for i in range(self.step):
            adv_imgs.requires_grad = True
            outputs, features = self.model(norm_fn(adv_imgs, m, s))

            loss = self.criterion_CE(outputs, labels)

            grad = torch.autograd.grad(loss, adv_imgs)[0]
            # compute grad mean
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad /= grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad += momentum*1.0 # decay term
            momentum = grad

            adv_imgs = adv_imgs.detach()+self.alpha*grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()

        return adv_imgs.detach(), labels
