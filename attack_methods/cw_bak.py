"""[summary]
"""
import torch
import torch.nn as nn
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
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        # delta = torch.zeros_like(imgs).to(imgs.device)
        delta = self.arctanh(imgs).detach()
        delta.requires_grad = True
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        best_adv_imgs = imgs.clone().detach()
        best_l2 = 1e6*torch.ones((len(imgs))).to(self.device)
        prev_loss = 1e6
        dim = len(imgs.size())

        criterion_mse = nn.MSELoss(reduction='none')
        flatten = nn.Flatten()

        for step in range(self.n_iters):
            print('C&W Attack Progress: {:.2f}%'.format(
                (step+1)/self.n_iters*100), end='\r')

            adv_imgs = self.scaler(delta)
            outputs, _, _, _ = self.target_cls(adv_imgs)

            loss_f = torch.sum(self.c*self._f(outputs, labels))
            _mse = criterion_mse(flatten(adv_imgs), flatten(imgs)).sum(dim=1)
            loss_mse = _mse.sum()

            loss = loss_f + loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1-correct)*(best_l2 > _mse.detach())
            best_l2 = mask*_mse.detach() + (1-mask)*best_l2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_imgs = mask*adv_imgs.detach() + (1-mask)*best_adv_imgs

            if step % (self.n_iters // 10) == 0:
                if loss > prev_loss:
                   return best_adv_imgs.detach(), labels
                prev_loss = loss.item()

            return best_adv_imgs.detach(), labels

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
