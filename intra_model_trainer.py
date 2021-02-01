import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import network_initialization, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    get_m_s,
    norm,
    get_optim,
    Loss,
    InterLoss,
    IntraLoss,
    get_center
)
from attack_methods import pgd, fgsm
from tqdm import tqdm


class Trainer:
    def __init__(self, args):
        # dataloader
        self.train_loader, self.dev_loader, _ = get_dataloader(args)
        # model initialization
        self.model = network_initialization(args)
        # get mean and std for normalization
        self.m, self.s = get_m_s(args)

        self.save_path = os.path.join(args.save_path, args.dataset)
        os.makedirs(self.save_path, exist_ok=True)

        pretrained_path = os.path.join(self.save_path, 'restricted_model.pt')
        self.checkpoint = torch.load(pretrained_path)
        self.center = self.checkpoint["center"]

        # set criterion
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion = Loss(args.num_class, args.device, pre_center=self.center, phase=args.phase)

        # set logger path
        log_num = 0
        if args.adv_train:
            while os.path.exists(f"logger/proposed/intra_loss/{args.dataset}/adv_train/v{str(log_num)}"):
                log_num += 1
            self.writer = SummaryWriter(f"logger/proposed/intra_loss/{args.dataset}/adv_train/v{str(log_num)}")
        else:
            while os.path.exists(f"logger/proposed/intra_loss/{args.dataset}/v{str(log_num)}"):
                log_num += 1
            self.writer = SummaryWriter(f"logger/proposed/intra_loss/{args.dataset}/v{str(log_num)}")

    def training(self, args):
        model = self.model
        if args.adv_train:
            print("Train the model with adversarial examples")
            attack_func = getattr(pgd, "PGD")

        # load the model weights
        model.module.load_state_dict(self.checkpoint["model_state_dict"])

        # set optimizer & scheduler
        optimizer, scheduler = get_optim(
            model, args.lr
        )
        optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])

        # base model
        model_name = f"intra_model.pt"
        if args.adv_train:
            model_name = f"{model_name.split('.')[0]}_adv_train.pt"
        model_path = os.path.join(self.save_path, model_name)

        self.writer.add_text(tag="argument", text_string=str(args.__dict__))
        self.writer.close()
        best_loss = 1000
        current_step = 0
        dev_step = 0

        trn_loss_log = tqdm(total=0, position=2, bar_format='{desc}')
        dev_loss_log = tqdm(total=0, position=4, bar_format='{desc}')
        best_epoch_log = tqdm(total=0, position=5, bar_format='{desc}')
        outer = tqdm(total=args.epochs, desc="Epoch", position=0, leave=False)
        # Train target classifier
        for epoch in range(args.epochs):
            _dev_loss = 0.0
            train = tqdm(total=len(self.train_loader), desc="Steps", position=1, leave=False)
            dev = tqdm(total=len(self.dev_loader), desc="Steps", position=3, leave=False)
            for step, (inputs, labels) in enumerate(self.train_loader):
                model.train()
                current_step += 1

                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                if args.adv_train:
                    attacker = attack_func(model, args)
                    adv_imgs, adv_labels = attacker.__call__(inputs, labels, norm, self.m, self.s)
                    inputs = torch.cat((inputs, adv_imgs), 0)
                    labels = torch.cat((labels, adv_labels))
                inputs = norm(inputs, self.m, self.s)

                logit, features = model(inputs)
                ce_loss = self.criterion_CE(logit, labels)
                intra_loss = self.criterion(features, labels, True)

                loss = 0.05*ce_loss + 0.95*intra_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #################### Logging ###################
                trn_loss_log.set_description_str(
                    f"[TRN] Total Loss: {loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, Intra Loss: {intra_loss.item():.4f}"
                )
                train.update(1)

            if epoch % 10 == 0:
                self.writer.add_embedding(
                    features,
                    metadata=labels.data.cpu().numpy(),
                    label_img=inputs,
                    global_step=current_step,
                    tag="[TRN]Features",
                )
                self.writer.close()

            for idx, (inputs, labels) in enumerate(self.dev_loader):
                model.eval()
                dev_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                if args.adv_train:
                    adv_imgs, adv_labels = attacker.__call__(inputs, labels, norm, self.m, self.s)
                    inputs = torch.cat((inputs, adv_imgs), 0)
                    labels = torch.cat((labels, adv_labels))
                inputs = norm(inputs, self.m, self.s)

                with torch.no_grad():
                    logit, features = model(inputs)
                    ce_loss = self.criterion_CE(logit, labels)
                    intra_loss = self.criterion(features, labels, False)
                    loss = ce_loss + intra_loss

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss / (idx + 1)

                    dev_loss_log.set_description_str(
                        f"[DEV] Loss: {dev_loss:.4f}"
                    )

                    #################### Logging ###################
                    dev.update(1)
                    if idx % args.dev_interval == 0:
                        self.writer.add_scalar("dev/loss", dev_loss.item(), dev_step)
                        self.writer.close()

            if dev_loss < best_loss:
                best_epoch_log.set_description_str(
                    f"Best Epoch: {epoch} / {args.epochs} | Best Loss: {dev_loss}"
                )
                best_loss = dev_loss
                torch.save(
                    {
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "trained_epoch": epoch
                    },
                    model_path
                )

            # tensorboard logging
            self.writer.add_scalar(
                "train/loss", loss.item(), global_step=current_step
            )
            self.writer.add_scalar(
                "train/ce_loss", ce_loss.item(), global_step=current_step
            )
            self.writer.add_scalar(
                "train/intra_loss", intra_loss.item(), global_step=current_step
            )
            self.writer.close()
            if epoch % 10 == 0:
                self.writer.add_embedding(
                    features,
                    metadata=labels.data.cpu().numpy(),
                    label_img=inputs,
                    global_step=current_step,
                    tag="[DEV]Features",
                )
                self.writer.close()

            scheduler.step(dev_loss)
            outer.update(1)

            # save the last epoch model
            save_last_path = f"{model_path[:-3]}_last.pt"
            torch.save(
                {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "trained_epoch": epoch
                },
                save_last_path
            )