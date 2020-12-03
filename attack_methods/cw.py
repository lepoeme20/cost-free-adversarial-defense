"""[summary]
"""
import torch
from attacks import Attack

class CW(Attack):
    def __init__(self, target_cls, args):
        super(CW, self).__init__("CW", target_cls)
        self.targeted = args.cw_targeted
        self.c = args.cw_c
        self.kappa = args.cw_kappa
        self.n_iters = args.cw_iters
        self.lr = args.cw_lr
        self.args = args

    def forward(self, imgs, labels, norm_fn, m, s):
        imgs = imgs.to(self.args.device)
        labels = labels.to(self.args.device)

        x_arctanh = self.arctanh(imgs)
        prev_loss = 1e6

        for step in range(self.n_iters):
            delta = torch.zeros_like(imgs).to(self.args.device)
            delta.detach_()
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta], lr=self.lr)

            print('C&W Attack Progress: {:.2f}%'.format(
                (step+1)/self.n_iters*100), end='\r')
            optimizer.zero_grad()
            adv_examples = self.scaler(x_arctanh + delta)
            loss1 = torch.sum(self.c*self._f(adv_examples, labels))
            loss2 = torch.functional.F.mse_loss(adv_examples, imgs, reduction='sum')

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            if step % (self.n_iters // 10) == 0:
                if loss > prev_loss:
                    break

                prev_loss = loss

            adv_imgs = self.scaler(x_arctanh + delta)
            return adv_imgs, labels

    def _f(self, adv_imgs, labels):
        outputs, _ = self.target_cls(adv_imgs)
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