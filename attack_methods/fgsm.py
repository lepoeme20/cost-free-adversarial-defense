"""
FGSM

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
from attacks import Attack

class FGSM(Attack):
    def __init__(self, target_cls, args):
        super(FGSM, self).__init__("FGSM", target_cls)
        self.eps = args.eps
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, imgs, labels, norm_fn, m, s):
        imgs = imgs.clone().detach()
        labels = labels

        imgs.requires_grad = True

        outputs, _ = self.target_cls(norm_fn(imgs, m, s))
        loss = self.criterion(outputs, labels)
        loss.backward()

        grad = imgs.grad.data

        adv_imgs = imgs+(self.eps*grad.sign())
        adv_imgs = torch.clamp(adv_imgs, 0, 1)

        return adv_imgs.detach(), labels
