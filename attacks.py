"""Adversarial attack class
"""
import os
import torch
from utils import get_m_s, norm

class Attack(object):
    """Base class for attacks

    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, attack_type, target_cls, img_type='float'):
        self.attack_name = attack_type
        self.target_cls = target_cls

        self.device = next(target_cls.parameters()).device

        self.mode = img_type

    def forward(self, *args):
        """Call adversarial examples
        Should be overridden by all attack classes
        """
        raise NotImplementedError

    def inference(self, args, save_path, file_name, data_loader):
        """[summary]

        Arguments:
            save_path {[type]} -- [description]
            data_loader {[type]} -- [description]
        """
        self.target_cls.eval()

        adv_list = []
        label_list = []

        correct = 0
        accumulated_num = 0.
        total_num = len(data_loader)
        m, s = get_m_s(args)

        for step, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(args.device)
            if imgs.size(1) == 1:
                imgs = imgs.expand(imgs.size(0), 3, imgs.size(2), imgs.size(3))
            labels = labels.to(args.device)
            adv_imgs, labels = self.__call__(imgs, labels, norm, m, s)
            adv_imgs = norm(adv_imgs, m, s)
            adv_list.append(adv_imgs.cpu())
            label_list.append(labels.cpu())

            accumulated_num += labels.size(0)

            if self.mode.lower() == 'int':
                adv_imgs = adv_imgs.float()/255.

            outputs, _ = self.target_cls(adv_imgs)
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()

            acc = 100 * correct / accumulated_num

            print('Progress : {:.2f}% / Accuracy : {:.2f}%'.format(
                (step+1)/total_num*100, acc), end='\r')

        print('Progress : {:.2f}% / Accuracy : {:.2f}%'.format(
            (step+1)/total_num*100, acc))

        adversarials = torch.cat(adv_list, 0)
        y = torch.cat(label_list, 0)

        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, file_name)
        torch.save((adversarials, y), save_path)

    def training(self, imgs, labels, m, s):
        adv_imgs, labels = self.__call__(imgs, labels, norm, m, s)
        return adv_imgs, labels

    def __call__(self, *args):
        self.target_cls.eval()
        adv_examples, labels = self.forward(*args)

        if self.mode.lower() == 'int':
            adv_examples, labels = (adv_examples*255).type(torch.uint8)

        return adv_examples, labels
