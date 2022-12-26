import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from utils import network_initialization, get_dataloader
from utils import (
    get_m_s,
    norm,
    get_optim,
    Loss,
    set_seed,
    get_center,
    LargeMarginLoss,
)
from tqdm import tqdm
import datetime
import time


class Trainer:
    def __init__(self, args):
        set_seed(args.seed)
        # dataloader
        self.train_loader, self.dev_loader, _ = get_dataloader(args)
        # model initialization
        self.model = network_initialization(args)
        # get mean and std for normalization
        self.m, self.s = get_m_s(args)

        self.save_path = os.path.join(args.save_path, args.dataset)
        os.makedirs(self.save_path, exist_ok=True)

        if args.resume:
            pretrained_path = os.path.join(
                self.save_path, f'{args.resume_model}.pt'
            )
        else:
            pretrained_path = os.path.join(
                self.save_path,
                f'restricted_model_{args.model}.pt'
            )
        self.checkpoint = torch.load(pretrained_path)
        self.model.module.load_state_dict(self.checkpoint["model_state_dict"])
        try:
            for param in self.model.module.fc.parameters():
                param.requires_grad = False
        except:
            for param in self.model.module.classifier.parameters():
                param.requires_grad = False
        self.center = get_center(
            self.model, self.train_loader, args.num_class, args.device, self.m, self.s
        )

        # set criterion
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion = Loss(
            args.num_class,
            args.device,
            pre_center=self.center,
            phase=args.phase,
        )

    def training(self, args):
        # set optimizer & scheduler
        optimizer, scheduler = get_optim(
            self.model, args.lr_intra, intra=True
        )

        # base model
        model_name = f"intra_model_{args.model}.pt"
        model_path = os.path.join(self.save_path, model_name)

        best_loss = 1000
        current_step = 0
        dev_step = 0

        if args.resume:
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])
            trained_epoch = self.checkpoint["trained_epoch"] + 1
            best_loss = self.checkpoint["best_loss"]
        else:
            trained_epoch = 0

        trn_loss_log = tqdm(total=0, position=2, bar_format='{desc}')
        dev_loss_log = tqdm(total=0, position=4, bar_format='{desc}')
        best_epoch_log = tqdm(total=0, position=5, bar_format='{desc}')
        outer = tqdm(total=args.epochs-trained_epoch, desc="Epoch", position=0, leave=False)

        os.makedirs('time_log', exist_ok=True)
        f = open(f"time_log/{args.dataset}_{args.phase}.txt", 'w')
        start_total = time.time()

        # train target classifier
        for epoch in range(trained_epoch, args.epochs):
            start_epoch = time.time()
            _dev_loss = 0.0
            train = tqdm(total=len(self.train_loader), desc="Steps", position=1, leave=False)
            dev = tqdm(total=len(self.dev_loader), desc="Steps", position=3, leave=False)

            for step, (inputs, labels) in enumerate(self.train_loader):
                self.model.train()
                current_step += 1

                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = norm(inputs, self.m, self.s)

                logit, features = self.model(inputs)
                _, predict = torch.max(logit, 1)
                correct_idx = predict.eq(labels)
                ce_loss = self.criterion_CE(logit, labels)
                intra_loss = self.criterion(features, labels, correct_idx)

                loss = 0.*ce_loss + 10.0*intra_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #################### Logging ###################
                trn_loss_log.set_description_str(
                    f"[TRN] Total Loss: {loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, Intra Loss: {intra_loss.item():.4f}"
                )
                train.update(1)

            epoch_time = round(time.time() - start_epoch)
            epoch_time = str(datetime.timedelta(seconds=epoch_time))
            f.write(f"Epoch {epoch+1}: "+str(epoch_time)+'\n')

            for idx, (inputs, labels) in enumerate(self.dev_loader):
                self.model.eval()
                dev_step += 1

                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = norm(inputs, self.m, self.s)

                with torch.no_grad():
                    logit, features = self.model(inputs)
                    _, predict = torch.max(logit, 1)
                    correct_idx = predict.eq(labels)

                    ce_loss = self.criterion_CE(logit, labels)
                    intra_loss = self.criterion(features, labels, correct_idx)
                    loss = intra_loss

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss / (idx + 1)

                    dev_loss_log.set_description_str(
                        f"[DEV] CE Loss: {ce_loss.item():.4f} Intra Loss: {intra_loss.item():.4f}"
                    )

                    dev.update(1)

            if dev_loss > best_loss:
                best_epoch_log.set_description_str(
                    f"Best Epoch: {epoch} / {args.epochs} | Best Loss: {dev_loss}"
                )
                best_loss = dev_loss
                torch.save(
                    {
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "trained_epoch": epoch,
                        "best_loss": dev_loss,
                        "center": self.center
                    },
                    f"{model_path.split('_')[0]}_{str(epoch)}_{'_'.join(model_path.split('_')[1:])}"
                )

            scheduler.step(dev_loss)
            outer.update(1)

        total_time = round(time.time() - start_total)
        total_time = str(datetime.timedelta(seconds=total_time))
        f.write(f"Total: "+str(total_time)+'\n')
        f.close()
