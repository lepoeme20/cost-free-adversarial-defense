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
    def __init__(self, target_cls, args, train_loader, blackbox_cls=None):
        super(MIM, self).__init__("MIM", target_cls, blackbox_cls)
        self.eps = args.eps
        self.alpha = args.eps/args.mim_step
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

        perturbation = 0
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

            grad = torch.autograd.grad(loss, adv_imgs)[0]
            # compute grad mean
            adv_mean = torch.mean(torch.abs(grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            grad /= adv_mean
            perturbation += grad

            adv_imgs = adv_imgs+(self.alpha*perturbation.sign())
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        return adv_imgs.detach(), labels
