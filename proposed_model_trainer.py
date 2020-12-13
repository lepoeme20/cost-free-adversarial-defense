import os
import torch
import torch.nn as nn
from utils import network_initialization, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    get_m_s,
    norm,
    get_optim,
    get_uniform_loss,
    AngleLoss,
    ProximityLoss,
)
from attack_methods import pgd


class ProposedTrainer:
    def __init__(self, args):
        # dataloader
        self.train_loader, self.dev_loader, _ = get_dataloader(args)
        # model initialization
        self.model = network_initialization(args)
        # get mean and std for normalization
        self.m, self.s = get_m_s(args)

        # check adversarial training
        if args.adv_training:
            root_path = os.path.join(args.save_path, "w_adv_training")
        else:
            root_path = os.path.join(args.save_path, "wo_adv_training")

        save_path = os.path.join(root_path, args.dataset)
        self.save_path = os.path.join(
            save_path, args.network, str(args.lr), str(args.batch_size), args.v
        )
        os.makedirs(self.save_path, exist_ok=True)

        # set criterion
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_MSE = nn.MSELoss()
        self.criterion_A_Softmax = AngleLoss()
        self.criterion_proximity = ProximityLoss(args.num_class)

        # set logger path
        log_num = 0
        while os.path.exists(f"logger/proposed/{str(log_num)}"):
            log_num += 1
        self.writer = SummaryWriter(f"logger/proposed/{str(log_num)}")

    def training(self, args):
        model = self.model

        # set optimizer & scheduler
        optimizer, scheduler = get_optim(model, args.lr)
        optimizer_prox = torch.optim.SGD(self.criterion_proximity.parameters(), lr=0.5)

        model_path = os.path.join(self.save_path, "proposed_model_100.pth")

        self.writer.add_text(tag="argument", text_string=str(args.__dict__))
        self.writer.close()
        best_loss = 1000
        current_step = 0
        dev_step = 0
        center = torch.rand(
            size=(args.num_class, 256),
            dtype=torch.float32,
            device=args.device,
            requires_grad=False,
        )

        # Train target classifier
        for epoch in range(args.epochs):
            _dev_loss = 0.0

            print("")
            for step, (inputs, labels) in enumerate(self.train_loader, 0):
                model.train()
                current_step += 1

                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = norm(inputs, self.m, self.s)

                # features: 256d
                logit, _, features, _ = model(inputs)
                ce_loss = self.criterion_CE(logit, labels)
                # ce_loss = self.criterion_A_Softmax(logit, labels)

                center, uniform_loss = get_uniform_loss(
                    center, features, labels, args.num_class
                )
                # new_labels = center[labels]
                # mse_loss = self.criterion_MSE(features, new_labels)
                mse_loss = self.criterion_proximity(center, features, labels)
                loss = ce_loss + 2 * uniform_loss + mse_loss
                optimizer.zero_grad()
                optimizer_proximity.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_proximity.step()
                #################### Logging ###################
                print(
                    f"[Trn] Epoch {epoch+1}/{args.epochs} Batch {step+1}/{len(self.train_loader)} Total Loss {loss.item():.4f} CE Loss {ce_loss.item():.4f} U Loss {uniform_loss.item():.4f} MSE Loss {mse_loss.item():.4f}",
                    end="\r",
                )

                if step == 0 or current_step % len(self.train_loader) * 30 == 0:
                    self.writer.add_scalar(
                        "train/loss", loss.item(), global_step=current_step
                    )
                    self.writer.close()
                    self.writer.add_embedding(
                        center,
                        metadata=list(range(0, args.num_class)),
                        global_step=current_step,
                        tag="Centers",
                    )
                    self.writer.close()
                    self.writer.add_embedding(
                        features,
                        metadata=labels.data.cpu().numpy(),
                        label_img=inputs,
                        global_step=current_step,
                        tag="Features",
                    )
                    self.writer.close()

            print("")
            for idx, (inputs, labels) in enumerate(self.dev_loader, 0):
                model.eval()
                dev_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = norm(inputs, self.m, self.s)

                with torch.no_grad():
                    logit, _, features, _ = model(inputs)
                    ce_loss = self.criterion_CE(logit, labels)
                    center, uniform_loss = get_uniform_loss(
                        center, features, labels, args.num_class
                    )
                    # new_labels = center[labels]
                    # mse_loss = self.criterion_MSE(features, new_labels)
                    mse_loss = get_proximity_loss(
                        center, features, labels, args.num_class
                    )
                    loss = ce_loss + uniform_loss + mse_loss

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss / (idx + 1)

                    print(
                        "[Dev] {}/{} Loss: {:.3f}".format(
                            idx + 1, len(self.dev_loader), dev_loss
                        ),
                        end="\r",
                    )

                    #################### Logging ###################
                    if idx % args.dev_interval == 0:
                        self.writer.add_scalar("dev/loss", loss.item(), dev_step)
                        self.writer.close()

            scheduler.step(dev_loss)

            if dev_loss < best_loss:
                best_loss = dev_loss
                self._save_model(model, optimizer, scheduler, epoch, model_path)

    def _save_model(self, model, optimizer, scheduler, epoch, path):
        print("The best model is saved")
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "trained_epoch": epoch,
            },
            path,
        )


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
