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

    def forward(self, images, labels, norm_fn, m, s):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(norm_fn(images, m, s)).detach()
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
            f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c*f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() - (1-mask)*best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.n_iters//10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images, labels
                prev_cost = cost.item()

        return best_adv_images, labels

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # return self.atanh(x*2-1)
        return torch.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(1*(i-j), min=-self.kappa)
