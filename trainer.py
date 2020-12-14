import os
import torch
import torch.nn as nn
from utils import network_initialization, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import get_m_s, norm, get_optim
from attack_methods import pgd

class Trainer():
    def __init__(self,args):
        # dataloader
        self.train_loader, self.dev_loader, _ = get_dataloader(args)
        # model initialization
        self.model = network_initialization(args)
        # get mean and std for normalization
        self.m, self.s = get_m_s(args)

        # check adversarial training
        if args.adv_training:
            root_path = os.path.join(args.save_path, 'w_adv_training')
        else:
            root_path = os.path.join(args.save_path, 'wo_adv_training')

        save_path = os.path.join(root_path, args.dataset)
        self.save_path = os.path.join(save_path, args.network, str(args.lr), str(args.batch_size), args.v)
        os.makedirs(self.save_path, exist_ok=True)

        # set criterion
        self.criterion_CE = nn.CrossEntropyLoss()

        # set logger path
        self.writer = SummaryWriter('logger/ce_loss')

    def training(self, args):
        model = self.model

        # set optimizer & scheduler
        optimizer, scheduler = get_optim(model, args.lr)

        model_path = os.path.join(self.save_path, "pretrained_model.pth")
        self.writer.add_text(tag='argument', text_string=str(args.__dict__))
        self.writer.close()
        best_loss = 1000
        current_step = 0
        dev_step = 0

        # Train target classifier
        for epoch in range(args.epochs):
            _dev_loss = 0.0

            print("")
            for step, (inputs, labels) in enumerate(self.train_loader, 0):
                model.train()
                current_step += 1

                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = norm(inputs, self.m, self.s)

                # Cross entropy loss
                logit, _, _, _ = model(inputs)
                loss = self.criterion_CE(logit, labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                #################### Logging ###################
                print("[Training] Epoch {}/{} Batch {}/{} Loss: {:.4f}".format(
                    epoch, args.epochs, step, len(self.train_loader), loss.item()
                ), end='\r')
                if step == 0 or step % args.sample_interval == 0:
                    self.writer.add_scalar('train/loss', loss.item(), current_step)
                    self.writer.close()

            print("")
            for idx, (inputs, labels) in enumerate(self.dev_loader, 0):
                model.eval()
                dev_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = norm(inputs, self.m, self.s)

                with torch.no_grad():
                    logit, _, _, _ = model(inputs)

                    # Cross entropy loss
                    loss = self.criterion_CE(logit, labels)

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss/(idx+1)

                    print('[Dev] {}/{} Loss: {:.3f}'.format(
                        idx+1, len(self.dev_loader), dev_loss), end='\r')

                    #################### Logging ###################
                    if idx % args.dev_interval == 0:
                        self.writer.add_scalar('dev/loss', loss.item(), dev_step)
                        self.writer.close()

            scheduler.step(dev_loss)

            if dev_loss < best_loss:
                best_loss = dev_loss
                self._save_model(model, optimizer, scheduler, epoch, model_path)


    def _save_model(self, model, optimizer, scheduler, epoch, path):
        print("The best model is saved")
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'trained_epoch': epoch,
        }, path)

    def _compute_mean(self, args, model):
        # compute total mean vector and class mean vector
        # construct empty tensors
        vector_size = 512
        class_mean = torch.zeros((args.num_class, vector_size), device=args.device)
        class_num = torch.zeros((args.num_class, 1), device=args.device)

        # feed whole w/o augmentation training set to the model for compute class mean
        for step, (inputs, labels) in enumerate(self.wo_aug_trn_loader, 0):
            model.eval()
            # inputs: [batch size, channels, *(image size)]
            # labels: [batch size]
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            inputs = norm(inputs, self.m, self.s)
            with torch.no_grad():
                logit, feature_vector = model(inputs)
                _, predicted = torch.max(logit, 1)

                for i in range(inputs.size(0)):
                    if predicted[i].eq(labels[i]):
                        # Add output feature vector to coresponding class mean vector
                        class_mean[labels[i]] += feature_vector[i]
                        # Compute number of each class
                        class_num[labels[i]] += 1

        # Compute class mean vectors
        class_mean /= class_num

        # Compute total mean vector
        total_mean = torch.mean(class_mean, 0)
        return class_mean, total_mean

    # for Adversarial Training
#    def _get_adv_imgs(self, args, model, imgs, labels):
#        # get pgd attack module
#        attack_module = globals()['pgd']
#        attack_func = getattr(attack_module, "PGD")
#        attacker = attack_func(model, args)
#        # get adversarial images
#
#        adv_imgs, adv_labels = attacker.training(imgs, labels, self.m, self.s)
#        adv_imgs = adv_imgs.detach()
#        imgs = torch.cat((imgs, adv_imgs), dim=0)
#        labels = torch.cat((labels, adv_labels), dim=0)
#
#        # shuffle
#        shuffle_idx = torch.randperm(labels.size(0)).to(labels.device)
#        imgs = imgs[shuffle_idx, ...]
#        labels = labels[shuffle_idx, ...]
#        model.train()
#
#        return imgs, labels
