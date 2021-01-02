"""
Iterative FGSM

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
from attacks import Attack

class BIM(Attack):
    def __init__(self, target_cls, args):
        super(BIM, self).__init__("BIM", target_cls)
        self.eps = args.eps
        self.alpha = args.eps/args.bim_step
        self.step = args.bim_step
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, imgs, labels, norm_fn, m, s):
        adv_imgs = imgs.clone().detach()
        adv_imgs.requires_grad = True

        for i in range(self.step):
            outputs, _ = self.target_cls(norm_fn(adv_imgs, m, s))
            loss = self.criterion(outputs, labels)

            grad = torch.autograd.grad(loss, adv_imgs)[0]
            adv_imgs = adv_imgs+(self.alpha*grad.sign())
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        return adv_imgs.detach(), labels
