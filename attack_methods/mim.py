"""
Iterative FGSM

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
from attacks import Attack

class MIM(Attack):
    def __init__(self, target_cls, args):
        super(MIM, self).__init__("MIM", target_cls)
        self.eps = args.eps
        self.alpha = args.eps/args.mim_step
        self.step = args.bim_step
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, imgs, labels, norm_fn, m, s):
        adv_imgs = imgs.clone().detach()
        adv_imgs.requires_grad = True

        perturbation = 0
        for i in range(self.step):
            outputs, _ = self.target_cls(norm_fn(adv_imgs, m, s))
            loss = self.criterion(outputs, labels)

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
