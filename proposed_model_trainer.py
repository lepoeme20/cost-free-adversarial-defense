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
    Loss
)
from attack_methods import pgd, fgsm
from tqdm import tqdm


class ProposedTrainer:
    def __init__(self, args):
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
        # self.criterion = Loss(args.num_class, 256, args.device)
        # Ablation study
        self.criterion = Loss(args.num_class, 256, args.device, args.intra_p, args.inter_p, args.adv_train)

        # set logger path
        log_num = 0
        if args.adv_train:
            while os.path.exists(f"logger/proposed/{args.dataset}/adv_train/v{str(log_num)}"):
                log_num += 1
            self.writer = SummaryWriter(f"logger/proposed/{args.dataset}/adv_train/v{str(log_num)}")
        else:
            while os.path.exists(f"logger/proposed/{args.dataset}/v{str(log_num)}"):
                log_num += 1
            self.writer = SummaryWriter(f"logger/proposed/{args.dataset}/v{str(log_num)}")

    def training(self, args):
        model = self.model
        if args.adv_train:
            print("Train the model with adversarial examples")
            attack_func = getattr(pgd, "PGD")

        pretrained_path = os.path.join(self.save_path, 'pretrained_model.pt')
        # load the model weights
        checkpoint = torch.load(pretrained_path)#, map_location=f"cuda:{args.device_ids[0]}")
        model.module.load_state_dict(checkpoint["model_state_dict"])

        # set optimizer & scheduler
        optimizer, scheduler = get_optim(model, args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_proposed = torch.optim.SGD(self.criterion.parameters(), lr=args.lr_proposed)
        scheduler_proposed = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_proposed, mode="min", factor=0.1, patience=20
        )

        # base model
        # model_name = f"proposed_model.pt"
        # ablation study
        model_name = f"proposed_model_intra_p_{args.intra_p}_inter_p_{args.inter_p}.pt"
        # model_name = f"proposed_model_intra_l_{args.lambda_intra}_inter_l_{args.lambda_inter}.pt"
        if args.adv_train:
            _model_name = model_name.split('.')[0]
            model_name = f"{_model_name}_adv_train.pt"
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
                if args.adv_train:
                    attacker = attack_func(model, args)
                    adv_imgs, adv_labels = attacker.__call__(inputs, labels, norm, self.m, self.s)
                    inputs = torch.cat((inputs, adv_imgs), 0)
                    labels = torch.cat((labels, adv_labels))
                inputs = norm(inputs, self.m, self.s)

                # features: 256d
                logit, _, features, _ = model(inputs)
                ce_loss = self.criterion_CE(logit, labels)
                intra_loss, inter_loss, center = self.criterion(features, labels)
                #TODO: Ablation study about lambda
                loss = ce_loss + args.lambda_intra*intra_loss + args.lambda_inter*inter_loss

                optimizer.zero_grad()
                optimizer_proposed.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_proposed.step()
                #################### Logging ###################
                trn_loss_log.set_description_str(
                    f"[TRN] Total Loss: {loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, Inter Loss: {inter_loss.item():.4f}, Intra Loss: {intra_loss.item():.4f}"
                )
                train.update(1)

            for idx, (inputs, labels) in enumerate(self.dev_loader):
                model.eval()
                dev_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if args.adv_train:
                    adv_imgs, adv_labels = attacker.__call__(inputs, labels, norm, self.m, self.s)
                    inputs = torch.cat((inputs, adv_imgs), 0)
                    labels = torch.cat((labels, adv_labels))
                inputs = norm(inputs, self.m, self.s)

                with torch.no_grad():
                    logit, _, features, _ = model(inputs)
                    ce_loss = self.criterion_CE(logit, labels)
                    intra_loss, inter_loss, center = self.criterion(features, labels)
                    loss = ce_loss + intra_loss + inter_loss

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss / (idx + 1)

                    dev_loss_log.set_description_str(
                        f"[DEV] Loss: {dev_loss:.4f}"
                    )

                    #################### Logging ###################
                    dev.update(1)
                    if idx % args.dev_interval == 0:
                        self.writer.add_scalar("dev/loss", loss.item(), dev_step)
                        self.writer.close()

            if dev_loss < best_loss:
                best_epoch_log.set_description_str(
                    f"Best Epoch: {epoch} / {args.epochs} | Best Loss: {dev_loss}"
                )
                best_loss = dev_loss
                torch.save(
                    {
                        "model_state_dict": model.module.state_dict(),
                        "center_state_dict": self.criterion.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "optimizer_proposed_state_dict": optimizer_proposed.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scheduler_proposed_state_dict": scheduler_proposed.state_dict(),
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
            self.writer.add_scalar(
                "train/inter_loss", inter_loss.item(), global_step=current_step
            )
            self.writer.close()
            if epoch % 10 == 0:
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

            scheduler.step(dev_loss)
            scheduler_proposed.step(dev_loss)
            outer.update(1)
