import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import network_initialization, get_dataloader
from tensorboardX import SummaryWriter
from utils import get_m_s, norm, get_optim


# increase the distance between total mean vector and each class mean vector
def fine_tuning(args):
    # set model
    model = network_initialization(args)
    train_loader, dev_loader, _ = get_dataloader(args)
    m, s = get_m_s(args)

    if args.adv_training:
        root_path = os.path.join(args.save_path, 'w_adv_training')
    else:
        root_path = os.path.join(args.save_path, 'wo_adv_training')
    pretrained_path = os.path.join(
        root_path, args.dataset, args.network, str(args.lr), str(args.batch_size), args.v
        )
    save_path = os.path.join(pretrained_path, "defense_model.pth")

    best_loss = 1e10
    total_step = 0
    dev_step = 0

    # set optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

    # load the best model
    checkpoint = torch.load(os.path.join(pretrained_path, "pretrained_model.pth"))
    model.module.load_state_dict(checkpoint['model_state_dict'])

    for epoch in range(args.finetune_epochs):
        _dev_loss = 0.0
        # Train
        log_name = '[Fine-tuning: orth]'
        print("")
        for step, (inputs, labels) in enumerate(train_loader, 0):
            model.train()
            total_step += 1

            inputs, labels = inputs.to(args.device), labels.to(args.device)

            if args.adv_training:
                inputs, labels = _get_adv_imgs(args, model, inputs, labels)
            inputs = norm(inputs, m, s)

            # logit, feature_128, feature_256, feature_1024 = model(inputs)
            # _, predicted = torch.max(logit, 1)

            # # init class mean and class num on every batch
            # class_mean = torch.zeros((args.num_class, 256), device=args.device)
            # class_num = torch.zeros((args.num_class, 1), device=args.device)
            # new_labels = torch.zeros_like(feature_vector)

            # for i in range(inputs.size(0)):
            #     if predicted[i].eq(labels[i]):
            #         # Add output feature vector to coresponding class mean vector
            #         class_mean[labels[i]] += feature_vector[i]
            #         # Compute number of each class
            #         class_num[labels[i]] += 1

            # # Compute class mean vectors
            # one = torch.ones(1, device=args.device)
            # zero = torch.zeros(1, device=args.device)
            # class_num = torch.where(class_num != zero, class_num, one)
            # class_mean /= class_num

            # #각 행에 true label에 따른 class-mean vector를 넣어줌
            # for i in range(feature_vector.size(0)):
            #     new_labels[i] = class_mean[labels[i]]

            # # (class_num x feature_size) * (feature_size x class_num)
            # orth = torch.matmul(class_mean, class_mean.transpose(0, 1))

            # # loss_orth = criterion_MSE(orth, torch.eye(orth.size(0), device=args.device))
            # loss_orth = criterion_COS(orth, torch.eye(orth.size(0), device=args.device))
            # loss_orth = torch.mean(loss_orth)
            # loss_dist = criterion_MSE(feature_vector, new_labels)
            # loss_ce = criterion_CE(logit, labels)

            # loss = loss_orth + loss_dist + loss_ce
            loss = _compute_loss(model, inputs, labels, args.num_class, args.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #################### Logging ###################
            print(
                log_name + f" Epoch {epoch+1}/{args.finetune_epochs} Batch {step}/{len(train_loader)} Loss: {loss.item():.4f}".format(
            ), end='\r')
            # if step == 0 or step % args.sample_interval == 0:
            #     self.tensorboard.add_scalar('train/loss', loss.item(), self.total_step)

        # Validation
        print("")
        for idx, (inputs, labels) in enumerate(dev_loader, 0):
            model.eval()
            dev_step += 1
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if args.adv_training:
                inputs, labels = _get_adv_imgs(args, model, inputs, labels)
            inputs = norm(inputs, m, s)

            with torch.no_grad():
                # logit, feature_128, feature_256, feature_1024 = model(inputs)
                # _, predicted = torch.max(logit, 1)

                # # init class mean and class num on every batch
                # class_mean = torch.zeros((args.num_class,256), device=args.device)
                # class_num = torch.zeros((args.num_class, 1), device=args.device)
                # new_labels = torch.zeros_like(feature_vector)

                # for i in range(inputs.size(0)):
                #     if predicted[i].eq(labels[i]):
                #         # Add output feature vector to coresponding class mean vector
                #         class_mean[labels[i]] += feature_vector[i]
                #         # Compute number of each class
                #         class_num[labels[i]] += 1

                # # Compute class mean vectors
                # one = torch.ones(1, device=args.device)
                # zero = torch.zeros(1, device=args.device)
                # class_num = torch.where(class_num != zero, class_num, one)
                # class_mean /= class_num

                # #각 행에 true label에 따른 class-mean vector를 넣어줌
                # for i in range(feature_vector.size(0)):
                #     new_labels[i] = class_mean[labels[i]]

                # # (class_num x feature_size) * (feature_size x class_num)
                # orth = torch.matmul(class_mean, class_mean.transpose(0, 1))

                # # loss_orth = criterion_MSE(orth, torch.eye(orth.size(0), device=args.device))
                # loss_orth = criterion_COS(orth, torch.eye(orth.size(0), device=args.device))
                # loss_orth = torch.mean(loss_orth)
                # loss_dist = criterion_MSE(feature_vector, new_labels)
                # loss_ce = criterion_CE(logit, labels)

                # loss = loss_orth + loss_dist + loss_ce

                loss = _compute_loss(model, inputs, labels, args.num_class, args.device)
                # Loss
                _dev_loss += loss
                dev_loss = _dev_loss/(idx+1)

                # if phase == 1:
                print('[Dev] {}/{} Loss: {:.3f}'.format(
                idx+1, len(dev_loader), dev_loss), end='\r')
                #################### Logging ###################
            scheduler.step(dev_loss)

        if dev_loss < best_loss:
            best_loss = dev_loss
            _save_model(model, optimizer, scheduler, epoch, save_path)


def _save_model(model, optimizer, scheduler, epoch, path):
    print("The best model is saved")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'trained_epoch': epoch,
    }, path)

def _compute_mean(args, model, m, s):
    # compute total mean vector and class mean vector
    # construct empty tensors
    vector_size = 256
    class_mean = torch.zeros((args.num_class, vector_size), device=args.device)
    class_num = torch.zeros((args.num_class, 1), device=args.device)

    args.batch_size = 1
    loader, _, _ = get_dataloader(args)
    # feed whole w/o augmentation training set to the model for compute class mean
    for _, (inputs, labels) in enumerate(loader, 0):
        model.eval()
        # inputs: [batch size, channels, *(image size)]
        # labels: [batch size]
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        inputs = norm(inputs, m, s)
        # with torch.no_grad():
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

def _compute_loss(model, inputs, labels, numclass, device):
    criterion_MSE = nn.MSELoss()
    criterion_COS = nn.L1Loss()
    criterion_CE = nn.CrossEntropyLoss()
    logit, feature_128, feature_256, feature_1024 = model(inputs)
    _, predicted = torch.max(logit, 1)

    feature_vectors = [feature_128, feature_256, feature_1024]
    feature_sizes = [128, 256, 1024]
    for (feature_vector, feature_size) in zip(feature_vectors, feature_sizes):
        # init class mean and class num on every batch
        class_mean = torch.zeros((numclass, feature_size), device=device)
        class_num = torch.zeros((numclass, 1), device=device)
        new_labels = torch.zeros_like(feature_vector)

        for i in range(inputs.size(0)):
            if predicted[i].eq(labels[i]):
                # Add output feature vector to coresponding class mean vector
                class_mean[labels[i]] += feature_vector[i]
                # Compute number of each class
                class_num[labels[i]] += 1

        # Compute class mean vectors
        one = torch.ones(1, device=device)
        zero = torch.zeros(1, device=device)
        class_num = torch.where(class_num != zero, class_num, one)
        class_mean /= class_num

        #각 행에 true label에 따른 class-mean vector를 넣어줌
        for i in range(feature_vector.size(0)):
            new_labels[i] = class_mean[labels[i]]

        # (class_num x feature_size) * (feature_size x class_num)
        orth = torch.matmul(class_mean, class_mean.transpose(0, 1))

        # loss_orth = criterion_MSE(orth, torch.eye(orth.size(0), device=args.device))
        loss_orth = criterion_COS(orth, torch.eye(orth.size(0), device=device))
        loss_orth = torch.mean(loss_orth)
        loss_dist = criterion_MSE(feature_vector, new_labels)
        loss_ce = criterion_CE(logit, labels)

        loss = loss_orth + loss_dist + loss_ce

    return loss
