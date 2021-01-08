import os
import torch
import torch.nn as nn
from utils import network_initialization, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import get_m_s, norm, get_optim, Loss
from tqdm import tqdm


class Trainer():
    def __init__(self,args):
        # dataloader
        self.train_loader, self.dev_loader, _ = get_dataloader(args)
        # model initialization
        self.model = network_initialization(args)
        # get mean and std for normalization
        self.m, self.s = get_m_s(args)
        self.save_path = os.path.join(args.save_path, args.dataset)
        os.makedirs(self.save_path, exist_ok=True)
        # set criterion
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion = Loss(args.num_class, args.device, pre=True)
        # set logger path
        log_num = 0
        while os.path.exists(f"logger/ce_loss/{args.dataset}/v{str(log_num)}"):
            log_num += 1
        self.writer = SummaryWriter(f'logger/ce_loss/{args.dataset}/v{str(log_num)}')

    def training(self, args):
        model = self.model

        # set optimizer & scheduler
        optimizer, scheduler, optimizer_proposed, scheduler_proposed = get_optim(
            model, args.lr, self.criterion, args.lr_proposed
        )

        model_path = os.path.join(self.save_path, "pretrained_model_inter.pt")
        self.writer.add_text(tag='argument', text_string=str(args.__dict__))
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
            for step, (inputs, labels) in enumerate(self.train_loader, 0):
                model.train()
                current_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs = norm(inputs, self.m, self.s)

                # Cross entropy loss
                logit, features = model(inputs)
                ce_loss = self.criterion_CE(logit, labels)
                restricted_loss, inter_loss, center = self.criterion(features, labels)
                loss = ce_loss + restricted_loss - inter_loss

                optimizer.zero_grad()
                optimizer_proposed.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer_proposed.step()

                #################### Logging ###################
                trn_loss_log.set_description_str(
                    f"[TRN] Total Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}, Res: {restricted_loss.item():4f}, Inter: {inter_loss.item():.4f}"
                )
                train.update(1)

                # tensorboard logging
                if current_step == 1 or current_step % len(self.train_loader) == 0:
                    self.writer.add_scalar(
                        tag="[TRN] loss", scalar_value=loss.item(), global_step=current_step
                    )

                if current_step == 1 or current_step % (len(self.train_loader)*1) == 0:
                    self.writer.add_embedding(
                        features,
                        metadata=labels.data.cpu().numpy(),
                        label_img=inputs,
                        global_step=current_step,
                        tag="[TRN]Features",
                    )

            for idx, (inputs, labels) in enumerate(self.dev_loader, 0):
                model.eval()
                dev_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs = norm(inputs, self.m, self.s)

                with torch.no_grad():
                    logit, features = model(inputs)

                    # Cross entropy loss
                    ce_loss = self.criterion_CE(logit, labels)
                    restricted_loss, inter_loss, _ = self.criterion(features, labels)
                    loss = ce_loss + restricted_loss - inter_loss
                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss/(idx+1)
                    dev_loss_log.set_description_str(
                        f"[DEV] Loss: {dev_loss:.4f}"
                    )
                    #################### Logging ###################
                    dev.update(1)
                    if dev_step == 1 or dev_step % len(self.dev_loader) == 0:
                        self.writer.add_scalar(
                            tag="[DEV] loss", scalar_value=loss.item(), global_step=dev_step
                        )

                    if dev_step == 1 or dev_step % (len(self.dev_loader)*10) == 0:
                        self.writer.add_embedding(
                            features,
                            metadata=labels.data.cpu().numpy(),
                            label_img=inputs,
                            global_step=dev_step,
                            tag="[DEV]Features",
                        )

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
                        "optimizer_proposed_state_dict": optimizer_proposed.state_dict(),
                        "scheduler_proposed_state_dict": scheduler_proposed.state_dict(),
                        "trained_epoch": epoch,
                        "center": center
                    },
                    model_path
                )

            scheduler.step(dev_loss)
            scheduler_proposed.step(dev_loss)
            outer.update(1)
            torch.save(
                {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "optimizer_proposed_state_dict": optimizer_proposed.state_dict(),
                    "scheduler_proposed_state_dict": scheduler_proposed.state_dict(),
                    "trained_epoch": epoch,
                    "center": center
                },
                os.path.join(self.save_path, "pretrained_model_last.pt")
            )
