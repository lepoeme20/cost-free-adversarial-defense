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
        super(BIM, self).__init__("BIM", target_cls)
        self.eps = args.eps
        self.alpha = args.eps/args.bim_step
        self.step = args.bim_step
        self.criterion_CE = nn.CrossEntropyLoss()
        if blackbox_cls is not None:
            self.model = blackbox_cls
        else:
            self.model = target_cls
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
        imgs = imgs.clone().detach()
        orig_imgs = imgs.clone().detach()

        for i in range(self.step):
            imgs.requires_grad = True
            outputs, features = self.model(norm_fn(imgs, m, s))

            if self.adaptive:
                ce_loss = self.criterion_CE(outputs, labels)
                intra_loss = self.criterion(features, labels)
                loss = ce_loss + intra_loss
            else:
                loss = self.criterion_CE(outputs, labels)

            grad = torch.autograd.grad(
                loss, imgs, retain_graph=False, create_graph=False)[0]

            adv_imgs = imgs + self.alpha*grad.sign()
            # a = max(ori_images-eps, 0)
            a = torch.clamp(orig_imgs - self.eps, min=0)
            # b = max(adv_images, a) = max(adv_images, ori_images-eps, 0)
            b = (adv_imgs >= a).float()*adv_imgs + (adv_imgs < a).float()*a
            # c = min(ori_images+eps, b) = min(ori_images+eps, max(adv_images, ori_images-eps, 0))
            c = (b > orig_imgs+self.eps).float()*(orig_imgs+self.eps) + (b <= orig_imgs + self.eps).float()*b
            # images = max(1, c) = min(1, ori_images+eps, max(adv_images, ori_images-eps, 0))
            imgs = torch.clamp(c, max=1).detach()

        return adv_imgs.detach(), labels
