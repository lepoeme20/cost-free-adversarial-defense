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

class BIM(Attack):
    def __init__(self, target_cls, args, train_loader, blackbox_cls=None):
        super(BIM, self).__init__("BIM", target_cls, blackbox_cls)
        self.eps = args.eps
        self.alpha = args.eps/args.bim_step
        self.step = args.bim_step
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

        for i in range(self.step):
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

            grad = torch.autograd.grad(
                loss, adv_imgs, retain_graph=False, create_graph=False)[0]
            adv_imgs = adv_imgs+(self.alpha*grad.sign())
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        return adv_imgs.detach(), labels
