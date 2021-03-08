import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from resnet import resnet34, resnet18
from small_net import smallnet
import torch.nn.functional as F

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def network_initialization(args):
    if 'cifar' in args.dataset:
        net = resnet34(args.num_class)
    else:
        net = smallnet()

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
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        drop_last=True
    )
    devloader = torch.utils.data.DataLoader(
        devset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        drop_last=True
    )
    tstloader = torch.utils.data.DataLoader(
        tstset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu
    )

    return trainloader, devloader, tstloader


def get_m_s(args):
    if args.dataset.lower() == "mnist":
        m, s = [0.1307,], [0.3081,]
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


def get_optim(model, lr):
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )
    return optimizer, scheduler


def __get_transformer(args):
    # with data augmentation
    trn_transformer = transforms.Compose(
        [
            transforms.Pad(int(args.padding/2)),
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # transformer for testing (validation)
    dev_transformer = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
        ]
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


def get_center(model, data_loader, num_class, device, m, s, feature_dim):
    print("Compute class mean vectors")
    center = torch.zeros((num_class, feature_dim), device=device)
    label_count = torch.zeros((num_class, 1), device=device)
    model.eval()
    with torch.no_grad():
        for (imgs, labels) in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            # if imgs.size(1) == 1:
            #     imgs = imgs.repeat(1, 3, 1, 1)

            imgs = norm(imgs, m, s)
            _, features = model(imgs)

            batch_count = torch.zeros_like(label_count, device=labels.device)
            batch_center = torch.zeros_like(center, device=device)
            for feature, label_idx in zip(features, labels):
                batch_center[label_idx] += feature
                batch_count[label_idx] += 1
            label_count += batch_count

            center += batch_center
    return center/label_count

def get_fprime(model, data_loader, center, m, s):
    print("Compute F'")
    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in data_loader:
            c = center[labels]

            imgs = norm(imgs, m, s)
            _, features = model(imgs)
            dist_mat = torch.cdist(features, center, p=2)



class Loss(nn.Module):
    def __init__(self, num_class, device, phase, model, m, s, data_loader, pre_center=None):
        super(Loss, self).__init__()
        self.num_class = num_class
        self.classes = torch.arange(num_class, dtype=torch.long, device=device)
        self.zero = torch.zeros(1, dtype=torch.float, device=device)
        if pre_center == None:
            self.center = nn.Parameter(torch.randn((num_class, 512), device=device))
        else:
            self.center = pre_center.data.detach().to(device)
        self.phase = phase
        center_dist_mat = torch.cdist(
            self.center, self.center, p=2
        )
        threshold = (torch.mean(center_dist_mat)).detach()
        self.thres_inter = threshold*3
        # self.thres_rest = threshold/3
        self.thres_rest = torch.tensor(8, device=device)
        # self.thres_rest = torch.full((512, ), 3., device=device)
        self.mse_loss = nn.MSELoss()

        if phase == 'restricted'
            print("Compute F'")
            model.eval()
            self.f_prime = {}
            with torch.no_grad():
                for i, (imgs, labels) in enumerate(data_loader):
                    imgs, labels = imgs.to(device), labels.to(device)
                    c = self.center[labels]
                    imgs = norm(imgs, m, s)
                    _, features = model(imgs)
                    dist_mat = torch.cdist(features, self.center, p=2)
                    mask = labels.unsqueeze(1).eq(self.classes).squeeze()
                    dist = torch.sum((dist_mat*mask), 1)
                    multiplier = (self.thres_rest-dist)/dist
                    print((features-c).size())
                    print(multiplier.unsqueeze(1).size())

                    self.f_prime[i] = features+torch.matmul((features-c), multiplier.unsqueeze(1))

    def forward(self, features, labels, batch_idx, train=True):
        if self.phase == 'inter':
            # expansion_loss = self.expansion_loss(features, labels)
            inter_loss = self.inter_loss(features, labels, train)
            # expansion_loss = self.expansion_loss(features, labels, train)
            return inter_loss, self.center.data
        elif self.phase == 'restricted':
            # restricted_loss = self.expansion_loss(features, labels)
            restricted_loss = self.mse_loss(features, self.f_prime[batch_idx])
            return restricted_loss
        elif self.phase == 'intra':
            intra_loss = self.intra_loss(features, labels)
            # inter_loss = self.inter_loss(features, labels)
            return intra_loss#, inter_loss

    def intra_loss(self, features, labels):
        masked_dist_mat = self._get_masked_dist_mat(features, labels)
        loss = (masked_dist_mat).sum() / labels.size(0)

        return loss

    def inter_loss(self, features, labels):
        center = self.center.data
        label_counts = torch.zeros((self.num_class, 1), device=labels.device)
        for label_idx in labels:
            label_counts[label_idx] += 1

        # added
        for feature, label_idx in zip(features, labels):
            center[label_idx] += feature
        center /= label_counts.clamp(min=1)

        dist_mat = torch.cdist(
            center, center, p=2
        )
        dist_mat = torch.where(dist_mat < self.thres_inter, self.thres_inter-dist_mat, self.zero)
        inter_loss = torch.mean(dist_mat)

        return inter_loss

    def expansion_loss(self, features, labels):
        masked_dist_mat = self._get_masked_dist_mat(features, labels)

        dist = torch.sum(masked_dist_mat, 1) # row sum
        dist = torch.where(dist < self.thres_rest, self.thres_rest-dist, dist-self.thres_rest)

        loss = dist.sum()/dist.size(0)
        return loss

    def _get_masked_dist_mat(self, features, labels):
        center = self.center.clone().detach()
        dist_mat = torch.cdist(features, center, p=2)

        mask = labels.unsqueeze(1).eq(self.classes).squeeze()
        return (dist_mat*mask)
