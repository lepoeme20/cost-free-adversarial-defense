"""
FGSM

This code is written by Seugnwan Seo
"""
import os
import sys
import torch
import torch.nn as nn
from attacks import Attack
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import Loss, get_center, get_m_s

class FGSM(Attack):
    def __init__(self, target_cls, args):
        super(FGSM, self).__init__("FGSM", target_cls)
        self.eps = args.eps
        self.criterion_CE = nn.CrossEntropyLoss()
        self.model = target_cls

    def forward(self, imgs, labels, norm_fn, m, s):
        imgs = imgs.clone().detach()

        imgs.requires_grad = True

        outputs, features = self.model(norm_fn(imgs, m, s))
        _, predict = torch.max(outputs, 1)

        loss = self.criterion_CE(outputs, labels)

        grad = torch.autograd.grad(
            loss, imgs, retain_graph=False, create_graph=False)[0]
        # print(torch.count_nonzero(grad.sign()))
        adv_imgs = imgs+(self.eps*grad.sign())
        adv_imgs = torch.clamp(adv_imgs, 0, 1)

        return adv_imgs.detach(), labels
