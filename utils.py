import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from resnet110 import resnet
from resnet import resnet34, resnet18
from smallnet import small_resnet
import torch.nn.functional as F


def network_initialization(args):
    net = resnet34(args.num_class)

    # Using multi GPUs if you have
    if torch.cuda.device_count() > 0:
       net = nn.DataParallel(net, device_ids=args.device_ids)

    # change device to set device (CPU or GPU)
    net.to(args.device)

    return net


def get_dataloader(args):
    transformer = __get_transformer(args)
    dataset = __get_dataset_name(args)
    trn_loader, dev_loader, tst_loader = __get_loader(args, dataset, transformer)

    return trn_loader, dev_loader, tst_loader


def __get_loader(args, data_name, transformer):
    root = args.data_root_path
    data_path = os.path.join(root, args.dataset.lower())
    dataset = getattr(torchvision.datasets, data_name)

    # set transforms
    trn_transform, tst_transform = transformer
    # call dataset
    # normal training set
    trainset = dataset(
        root=data_path, download=True, train=True, transform=trn_transform
    )
    trainset, devset = torch.utils.data.random_split(
        trainset, [int(len(trainset) * 0.7), int(len(trainset) * 0.3)]
    )
    # validtaion, testing set
    tstset = dataset(
        root=data_path, download=True, train=False, transform=tst_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu
    )
    devloader = torch.utils.data.DataLoader(
        devset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu
    )
    tstloader = torch.utils.data.DataLoader(
        tstset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu
    )

    return trainloader, devloader, tstloader


def get_m_s(args):
    if args.dataset.lower() == "mnist":
        m, s = [0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]
    elif args.dataset.lower() == "cifar10":
        m, s = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif args.dataset.lower() == "cifar100":
        m, s = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
    elif args.dataset.lower() == "fmnist":
        m, s = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    return m, s


def norm(tensor, m, s):
    output = torch.rand_like(tensor)
    for c in range(output.size(1)):
        output[:, c, :, :] = (tensor[:, c, :, :] - m[c]) / s[c]
    return output


def get_optim(model, lr, criterion=None, proposed_lr=None):
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )
    if criterion != None:
        optimizer_proposed = optim.SGD(
            criterion.parameters(), lr=proposed_lr #, momentum=0.9, weight_decay=1e-3
        )
        scheduler_proposed = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_proposed, mode='min', factor=0.1, patience=10
        )
        return optimizer, scheduler, optimizer_proposed, scheduler_proposed
    else:
        return optimizer, scheduler


def __get_transformer(args):
    # with data augmentation
    trn_transformer = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.crop_size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # transformer for testing (validation)
    dev_transformer = transforms.Compose(
        [transforms.Resize(args.crop_size), transforms.ToTensor(),]
    )

    return trn_transformer, dev_transformer


def __get_dataset_name(args):
    if args.dataset.lower() == "mnist":
        d_name = "MNIST"
    elif args.dataset.lower() == "fmnist":
        d_name = "FashionMNIST"
    elif args.dataset.lower() == "cifar10":
        d_name = "CIFAR10"
    elif args.dataset.lower() == "cifar100":
        d_name = "CIFAR100"
    return d_name



class Loss(nn.Module):
    def __init__(self, num_class,device, pre_center=None, pre=False):
        super(Loss, self).__init__()
        self.num_class = num_class
        self.classes = torch.arange(num_class, dtype=torch.long, device=device)
        if pre_center == None:
            self.center = nn.Parameter(torch.randn((num_class, 512), device=device))
        else:
            self.center = pre_center.data.detach().to(device)
        self.pre = pre
        center_dist_mat = torch.cdist(
            self.center, self.center, p=2
        )
        self.threshold = (torch.mean(center_dist_mat) * 2).detach()
        self.empty_parameter = nn.Parameter(torch.tensor([1.]))

    def forward(self, features, labels):
        if self.pre:
            expansion_loss = self.expansion_loss(features, labels)
            inter_loss = self.inter_loss(features, labels)
            return expansion_loss, inter_loss, self.center
        else:
            intra_loss = self.intra_loss(features, labels)
            inter_loss = self.inter_loss(features, labels)
            return intra_loss, inter_loss

    def intra_loss(self, features, labels):
        batch_size = features.size(0)
        dist_mat = torch.cdist(
            features, self.center, p=2
        )

        mask = labels.unsqueeze(1).eq(self.classes).squeeze()

        loss = (dist_mat*mask).sum() / batch_size

        return loss

    def inter_loss(self, features, labels):
        count_input = torch.zeros((self.num_class, 1), device=labels.device)
        counts = torch.scatter_add(
            count_input, 0, labels.unsqueeze(1), torch.ones_like(features)
        )
        center = self.center.scatter_add(
            0, labels.unsqueeze(1).repeat(1, features.size(-1)), features
        )
        center /= counts.clamp(min=1)

        dist_mat = torch.cdist(
            center, center, p=2
        )
        zero = torch.zeros(1, dtype=torch.float, device=features.device)
        dist = torch.where(dist_mat < self.threshold, dist_mat, zero)
        loss = torch.mean(dist)
        # loss = dist_mat.sum() / (dist_mat.size(0) * dist_mat.size(1))
        # loss = torch.reciprocal(_loss)

        return loss

    def expansion_loss(self, features, labels):
        dist_mat = torch.cdist(
            features, self.center, p=2
        )

        mask = labels.unsqueeze(1).eq(self.classes).squeeze()
        masked_dist_mat = dist_mat * mask

        dist = torch.sum(masked_dist_mat, 1) # row sum
        dist = torch.where(dist < (self.threshold/3), (self.threshold/3)-dist, dist-(self.threshold/3))

        loss = (dist.sum()/labels.size(0))

        return loss

