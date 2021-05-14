import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from utils import network_initialization, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    get_m_s,
    norm,
    get_optim,
    Loss,
    set_seed,
    get_center,
    LargeMarginLoss,
)
from attack_methods import pgd, fgsm
from tqdm import tqdm


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
        # pretrained_path = os.path.join(self.save_path, 'intra_model.pt')
        self.checkpoint = torch.load(pretrained_path)
        self.model.module.load_state_dict(self.checkpoint["model_state_dict"])
        for param in self.model.module.fc.parameters():
            param.requires_grad = False
        dim = 120 if 'mnist' in args.dataset else 512
        self.center = get_center(
            self.model, self.train_loader, args.num_class, args.device, self.m, self.s, dim
        )

        # set criterion
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion = Loss(
            args.num_class,
            args.device,
            pre_center=self.center,
            phase=args.phase,
        )
        self.lm = LargeMarginLoss(
            gamma=10000,
            alpha_factor=4,
            top_k=1,
            dist_norm=2 # np.inf
        )

        # set logger path
        log_num = 0
        if args.adv_train:
            while os.path.exists(
                    f"logger/proposed/intra_loss_{args.model}/{args.dataset}/adv_train/v{str(log_num)}"
            ):
                log_num += 1
            self.writer = SummaryWriter(
                f"logger/proposed/intra_loss_{args.model}/{args.dataset}/adv_train/v{str(log_num)}"
            )
        else:
            while os.path.exists(
                    f"logger/proposed/intra_loss_{args.model}/{args.dataset}/v{str(log_num)}"
            ):
                log_num += 1
            self.writer = SummaryWriter(
                f"logger/proposed/intra_loss_{args.model}/{args.dataset}/v{str(log_num)}"
            )

    def training(self, args):
        if args.adv_train:
            print("Train the model with adversarial examples")
            attack_func = getattr(pgd, "PGD")

        # set optimizer & scheduler
        optimizer, scheduler = get_optim(
            self.model, args.lr_intra, intra=True
        )

        # base model
        model_name = f"intra_model_{args.model}.pt"
        if args.adv_train:
            model_name = f"{model_name.split('.')[0]}_adv_train.pt"
            print(model_name)
        model_path = os.path.join(self.save_path, model_name)

        self.writer.add_text(tag="argument", text_string=str(args.__dict__))
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

        # train target classifier
        for epoch in range(trained_epoch, args.epochs):
            _dev_loss = 0.0
            train = tqdm(total=len(self.train_loader), desc="Steps", position=1, leave=False)
            dev = tqdm(total=len(self.dev_loader), desc="Steps", position=3, leave=False)

            for step, (inputs, labels) in enumerate(self.train_loader):
                self.model.train()
                current_step += 1

                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if args.adv_train:
                    attacker = attack_func(self.model, args)
                    adv_imgs, adv_labels = attacker.__call__(inputs, labels, norm, self.m, self.s)
                    inputs = torch.cat((inputs, adv_imgs), 0)
                    labels = torch.cat((labels, adv_labels))
                inputs = norm(inputs, self.m, self.s)
                # one_hot = torch.zeros(
                #     (len(labels), args.num_class),
                #     device=labels.device
                # ).scatter_(1, labels.unsqueeze(1), 1.)

                logit, features = self.model(inputs)
                _, predict = torch.max(logit, 1)
                correct_idx = predict.eq(labels)
                ce_loss = self.criterion_CE(logit, labels)
                intra_loss = self.criterion(features, labels, correct_idx)
                # margin_loss = self.lm(logit, one_hot, features)

                loss = 0.00*ce_loss + 10.0*intra_loss # + margin_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #################### Logging ###################
                trn_loss_log.set_description_str(
                    f"[TRN] Total Loss: {loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, Intra Loss: {intra_loss.item():.4f}"
                )
                train.update(1)

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

            for idx, (inputs, labels) in enumerate(self.dev_loader):
                self.model.eval()
                dev_step += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if args.adv_train:
                    adv_imgs, adv_labels = attacker.__call__(inputs, labels, norm, self.m, self.s)
                    inputs = torch.cat((inputs, adv_imgs), 0)
                    labels = torch.cat((labels, adv_labels))
                inputs = norm(inputs, self.m, self.s)
                # one_hot = torch.zeros(
                #     (len(labels), args.num_class),
                #     device=labels.device
                # ).scatter_(1, labels.unsqueeze(1), 1.)

                with torch.no_grad():
                    logit, features = self.model(inputs)
                    _, predict = torch.max(logit, 1)
                    correct_idx = predict.eq(labels)

                    ce_loss = self.criterion_CE(logit, labels)
                    intra_loss = self.criterion(features, labels, correct_idx)
                    # margin_loss = self.lm(logit, one_hot, features)
                    loss = intra_loss # + margin_loss

                    # Loss
                    _dev_loss += loss
                    dev_loss = _dev_loss / (idx + 1)

                    dev_loss_log.set_description_str(
                        f"[DEV] CE Loss: {ce_loss.item():.4f} Intra Loss: {intra_loss.item():.4f}"
                    )

                    #################### Logging ###################
                    dev.update(1)
                    if dev_step == 1 or dev_step % len(self.dev_loader) == 0:
                        self.writer.add_scalar(
                            tag="[DEV] loss",
                            scalar_value=dev_loss.item(),
                            global_step=dev_step
                        )

                    if dev_step == 1 or dev_step % (len(self.dev_loader)*10) == 0:
                        self.writer.add_embedding(
                            features,
                            metadata=labels.data.cpu().numpy(),
                            label_img=inputs,
                            global_step=dev_step,
                            tag="[DEV] Features",
                        )

            # if epoch > 50 and dev_loss < best_loss:
            if dev_loss < best_loss:
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
                    # model_path[:-12] + str(epoch) + '_' + model_path[-12:]
                    f"{model_path.split('_')[0]}_{str(epoch)}_{'_'.join(model_path.split('_')[1:])}"
                    # model_path[:-8] + str(epoch) + '_' + model_path[-8:]
                    # model_path
                )

            scheduler.step(dev_loss)
            outer.update(1)
