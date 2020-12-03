"""[summary]

"""
import torch
import torch.nn as nn
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
        self.alpha = args.eps/4
        self.n_iters = args.pgd_iters
        self.random_start = args.pgd_random_start
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, imgs, labels, norm_fn, m, s):
        imgs = imgs.to(self.args.device)
        labels = labels.to(self.args.device)
        imgs.requires_grad = True

        if self.random_start:
            imgs = imgs + torch.empty_like(imgs).uniform_(-self.eps, self.eps)
            imgs = torch.clamp(imgs, 0, 1)

        for _ in range(self.n_iters):
            outputs, _ = self.target_cls(norm_fn(imgs, m, s))
            loss = self.criterion(outputs, labels)

            grad = torch.autograd.grad(loss, imgs)[0]

            adversarial_examples = imgs + self.alpha*grad.sign()
            eta = torch.clamp(adversarial_examples - imgs, min=-self.eps, max=self.eps)
            imgs = torch.clamp(imgs + eta, 0, 1)

        return imgs, labels