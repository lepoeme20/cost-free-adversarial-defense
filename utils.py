import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from resnet import resnet34
from resnet110 import resnet
import torch.nn.functional as F


def network_initialization(args):
    if args.network == "resnet110":
        net = resnet(args.num_class, args.rgb, args.batch_size)
    else:
        net = resnet34(args.num_class, args.rgb, args.batch_size)

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
        m, s = [0.1307], [0.3081]
    elif args.dataset.lower() == "cifar10":
        m, s = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    elif args.dataset.lower() == "cifar100":
        m, s = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
    elif args.dataset.lower() == "fmnist":
        m, s = [0.5], [0.5]

    return m, s


def norm(tensor, m, s):
    output = torch.rand_like(tensor)
    for c in range(output.size(1)):
        output[:, c, :, :] = (tensor[:, c, :, :] - m[c]) / s[c]
    return output


def get_optim(model, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=20
    )
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


def get_uniform_loss(center, features, labels, num_class):
    # Update center
    # selected_center = center[labels]

    # diff = selected_center - features

    # labels_rh = labels.reshape(-1, 1)

    # adj = labels_rh.eq(labels_rh.transpose(0, 1))

    # denom = torch.sum(adj, axis=1, keepdim=True)
    # diff /=denom

    # # eq. (7)
    # center.scatter_add_(0, labels_rh.repeat(1, diff.size(-1)), diff)
    # center.detach_()
    # center = torch.clamp(center, min=torch.min(features).item(), max=torch.max(features).item())

    # center[0] + new_diff[1] + new_diff[2]
    # center[1] + new_diff[0] + new_diff[3] + new_diff[4] + new_diff[5]

    count_input = torch.zeros((num_class, 1), device=labels.device)
    center.scatter_add_(0, labels.unsqueeze(1).repeat(1, features.size(-1)), features)
    counts = torch.scatter_add(
        count_input, 0, labels.unsqueeze(1), torch.ones_like(features)
    )
    center /= counts.clamp(min=1)

    dist_mat = torch.cdist(center, center, p=float("Inf"))
    combi = torch.combinations(torch.tensor(range(num_class)))

    total_dist = 0
    for class_idx in combi:
        i, j = class_idx
        dist = dist_mat[i][j]
        dist = torch.reciprocal(dist)
        total_dist += dist

    # uniform loss
    uniform_loss = (total_dist * 2) * (1 / (num_class * (num_class - 1)))
    return center.detach(), uniform_loss


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, inputs, target):
        self.it += 1
        cos_theta, phi_theta = inputs
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        # index = index.byte()
        index = index.to(dtype=torch.bool)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


# features = torch.rand((5, 5))
# labels = torch.randint(0, 10, (5, ))

# center = torch.zeros((10, 5))

# center.scatter_add_(0, labels.unsqueeze(1).repeat(1, features.size(-1)), features)
# counts = torch.scatter_add(torch.zeros((10, 1)), 0, labels.unsqueeze(1), torch.ones_like(features))
# center /= counts.clamp(min=1)


class ProximityLoss(nn.Module):
    def __init__(self, num_class):
        self.num_class = num_class

    def get_proximity_loss(center, features, labels):
        batch_size = features.size(0)
        # distmat = torch.add(
        #     torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_class),
        #     torch.pow(center, 2).sum(dim=1, keepdim=True).expand(num_class, batch_size).t(),
        # )
        # distmat = torch.addmm(distmat, features, center.t(), beta=1, alpha=-2)
        dist_mat = torch.cdist(features, center, p=2)
        classes = torch.arange(self.num_class, dtype=torch.long, device=features.device)

        mask = labels.unsqueeze(1).eq(classes).squeeze()

        dist = []
        for i in range(batch_size):
            value = dist_mat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e12)
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

