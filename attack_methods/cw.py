"""
C&W

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
import torch.optim as optim

from attacks import Attack

class CW(Attack):
    def __init__(self, target_cls, args):
        super(CW, self).__init__("CW", target_cls)
        self.targeted = args.cw_targeted
        self.c = args.cw_c
        self.kappa = args.cw_kappa
        self.n_iters = args.cw_iters

        self.lr = args.cw_lr
        self.device = args.device

    def forward(self, imgs, labels, norm_fn, m, s):
        images = imgs.clone().detach()
        labels = labels.clone().detach()

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.n_iters):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs, _ = self.target_cls(norm_fn(adv_images, m, s))
            f_Loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c*f_Loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.n_iters//10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images, labels
                prev_cost = cost.item()

        return best_adv_images, labels

    def _f(self, outputs, labels):
        y_onehot = torch.nn.functional.one_hot(labels)

        real = (y_onehot * outputs).sum(dim=1)
        other, _ = torch.max((1-y_onehot)*outputs, dim=1)

        if self.targeted:
            loss = torch.clamp(other-real, min=-self.kappa)
        else:
            loss = torch.clamp(real-other, min=-self.kappa)

        return loss

    def arctanh(self, imgs):
        scaling = torch.clamp(imgs, 0, 1)
        x = 0.999999 * scaling

        return 0.5*torch.log((1+x)/(1-x))

    def scaler(self, x_atanh):
        return ((torch.tanh(x_atanh))+1) * 0.5

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp((j-i), min=-self.kappa)
