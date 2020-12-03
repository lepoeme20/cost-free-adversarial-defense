import torch
import torch.nn as nn
from attacks import Attack

class BIM(Attack):
    """Reproduce Iterative Fast Gradients Sign Method (I-FGSM)
    in the paper 'Adversarial Examples in the Physical World'

    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        eps {float} -- Magnitude of perturbation
    """
    def __init__(self, target_cls, args):
        super(BIM, self).__init__("BIM", target_cls)
        self.eps = args.eps
        self.alpha = args.eps/args.bim_step
        self.step = args.bim_step
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, imgs, labels, norm_fn, m, s):
        imgs = imgs.to(self.args.device)
        labels = labels.to(self.args.device)

        adv_imgs = imgs.clone()
        adv_imgs.requires_grad = True

        for i in range(self.step):
            outputs, _ = self.target_cls(norm_fn(adv_imgs, m, s))
            loss = self.criterion(outputs, labels)

            gradients = torch.autograd.grad(loss, adv_imgs)[0]
            adv_imgs = adv_imgs+(self.eps*gradients.sign())
            adv_imgs = torch.clamp(adv_imgs, 0, 1)

        return adv_imgs, labels
