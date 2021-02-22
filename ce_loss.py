import os
import torch
import torch.nn as nn
from utils import network_initialization, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import get_m_s, norm, get_optim, set_seed
from tqdm import tqdm

class Trainer():
    def __init__(self,args):
        set_seed(args.seed)
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

        # set logger path
        log_num = 0
        while os.path.exists(f"logger/ce_{args.ce_epoch}/{args.dataset}/v{str(log_num)}"):
            log_num += 1
        self.writer = SummaryWriter(f'logger/ce_{args.ce_epoch}/{args.dataset}/v{str(log_num)}')

    def training(self, args):
        # set optimizer & scheduler
        optimizer, scheduler = get_optim(self.model, args.lr)

        model_path = os.path.join(self.save_path, f"ce_{args.ce_epoch}_model.pt")
        self.writer.add_text(tag='argument', text_string=str(args.__dict__))
        self.writer.close()
        best_loss = 1000
        current_step = 0
        dev_step = 0

        trn_loss_log = tqdm(total=0, position=2, bar_format='{desc}')
        dev_loss_log = tqdm(total=0, position=4, bar_format='{desc}')
        best_epoch_log = tqdm(total=0, position=5, bar_format='{desc}')
        outer = tqdm(total=args.ce_epoch, desc="Epoch", position=0, leave=False)

        # Train target classifier
        for epoch in range(args.ce_epoch):
            _dev_loss = 0.0
            train = tqdm(total=len(self.train_loader), desc="[TRN] Step", position=1, leave=False)
            dev = tqdm(total=len(self.dev_loader), desc="[DEV] Step", position=3, leave=False)
            for step, (inputs, labels) in enumerate(self.train_loader, 0):
                self.model.train()
                current_step += 1

                inputs, labels = inputs.to(args.device), labels.to(args.device)
                # if inputs.size(1) == 1:
                #     inputs = inputs.repeat(1, 3, 1, 1)
                inputs = norm(inputs, self.m, self.s)

                # Cross entropy loss
                logit, features = self.model(inputs)
                loss = self.criterion_CE(logit, labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()

                #################### Logging ###################
                trn_loss_log.set_description_str(
                    f"[TRN] Loss: {loss.item():.4f}"
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
                        tag="[TRN] Features",
                    )

            for idx, (inputs, labels) in enumerate(self.dev_loader, 0):
                self.model.eval()
                dev_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                # if inputs.size(1) == 1:
                #     inputs = inputs.repeat(1, 3, 1, 1)
                inputs = norm(inputs, self.m, self.s)

                with torch.no_grad():
                    logit, features = self.model(inputs)

                    # Cross entropy loss
                    loss = self.criterion_CE(logit, labels)

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss/(idx+1)


                    #################### Logging ###################
                    dev.update(1)
                    dev_loss_log.set_description_str(
                        f"[DEV] Loss: {dev_loss.item():.4f}"
                    )
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
                            tag="[DEV] Features",
                        )

            scheduler.step(dev_loss)
            outer.update(1)

            if dev_loss < best_loss:
                best_epoch_log.set_description_str(
                    f"Best Epoch: {epoch} / {args.ce_epoch} | Best Loss: {dev_loss}"
                )
                best_loss = dev_loss
                torch.save(
                    {
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "trained_epoch": epoch,
                    },
                    model_path
                )

