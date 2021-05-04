import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from resnet import resnet34, resnet18, resnet110
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
    if 'mnist' not in args.dataset:
        if args.model=='34':
            net = resnet34(args.num_class)
        elif args.model=='110':
            net = resnet110(args.num_class)
        else:
            net = resnet18(args.num_class)
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


class Loss(nn.Module):
    def __init__(self, num_class, device, phase, pre_center, dist=0):
        super(Loss, self).__init__()
        self.num_class = num_class
        self.classes = torch.arange(num_class, dtype=torch.long, device=device)
        self.center = pre_center.data.detach().to(device)
        self.phase = phase
        self.thres_rest = torch.tensor(dist, device=device)
        # center_dist_mat = torch.cdist(
        #      self.center, self.center, p=2
        #  )
        # print(center_dist_mat)

    def forward(self, features, labels):
        if self.phase == 'restricted':
            restricted_loss = self.expansion_loss(features, labels)
            return restricted_loss
        elif self.phase == 'intra':
            intra_loss = self.intra_loss(features, labels)
            return intra_loss

    def intra_loss(self, features, labels):
        masked_dist_mat = self._get_masked_dist_mat(features, labels)
        # loss = (masked_dist_mat).sum() / labels.size(0)
        dist = torch.sum(masked_dist_mat, 1)
        dist = torch.where(
            dist < self.thres_rest, dist, torch.zeros(1, dtype=torch.float32, device=features.device)
        )
        loss = dist.sum()/dist.size(0)

        return loss

    def expansion_loss(self, features, labels):
        masked_dist_mat = self._get_masked_dist_mat(features, labels)

        dist = torch.sum(masked_dist_mat, 1) # row sum
        # print(dist)
        # dist = torch.where(dist < self.thres_rest, self.thres_rest-dist, dist-self.thres_rest)
        dist = torch.where(dist < self.thres_rest, self.thres_rest-dist, torch.zeros(1, dtype=torch.float32, device=dist.device))

        loss = dist.sum()/dist.size(0)
        return loss

    def _get_masked_dist_mat(self, features, labels):
        center = self.center.clone().detach()
        dist_mat = torch.cdist(features, center, p=2)

        mask = labels.unsqueeze(1).eq(self.classes).squeeze()
        return (dist_mat*mask)

def _max_with_relu(a, b):
    return a + F.relu(b - a)

def _get_grad(out_, in_):
    grad, *_ = torch.autograd.grad(out_, in_,
                                   grad_outputs=torch.ones_like(out_, dtype=torch.float32),
                                   retain_graph=True)
    return grad.view(in_.shape[0], -1)

class LargeMarginLoss:
    """Large Margin Loss
    A Pytorch Implementation of `Large Margin Deep Networks for Classification`
    Referenced to Official TF Repo ( https://github.com/google-research/google-research/tree/master/large_margin )
    Docs is written with referenced to Official TF Repo

    Arguments :
          gamma (float): Desired margin, and distance to boundary above the margin will be clipped.
          alpha_factor (float): Factor to determine the lower bound of margin.
              Both gamma and alpha_factor determine points to include in training
              the margin these points lie with distance to boundary of [gamma * (1 - alpha), gamma]
          top_k (int):Number of top classes to include in the margin loss.
          dist_norm (1, 2, np.inf): Distance to boundary defined on norm
          epslion (float): Small number to avoid division by 0.
          use_approximation (bool):
          loss_type ("all_top_k", "worst_top_k", "avg_top_k"):  If 'worst_top_k'
              only consider the minimum distance to boundary of the top_k classes. If
              'average_top_k' consider average distance to boundary. If 'all_top_k'
              consider all top_k. When top_k = 1, these choices are equivalent.
    """
    def __init__(self,
                 gamma=10000.0,
                 alpha_factor=4.0,
                 top_k=1,
                 dist_norm=2,
                 epsilon=1e-8,
                 use_approximation=True,
                 loss_type="all_top_k"):

        self.dist_upper = gamma
        self.dist_lower = gamma * (1.0 - alpha_factor)

        self.alpha = alpha_factor
        self.top_k = top_k
        self.dual_norm = {1: np.inf, 2: 2, np.inf: 1}[dist_norm]
        self.eps = epsilon

        self.use_approximation = use_approximation
        self.loss_type = loss_type

    def __call__(self, logits, onehot_labels, feature_maps):
        """Getting Large Margin loss

        Arguments :
            logits (Tensor): output of Network before softmax
            onehot_labels (Tensor): One-hot shaped label
            feature_maps (list of Tensor): Target feature maps(Layer of NN) want to enforcing by Large Margin

        Returns :
            loss:  Large Margin loss
        """
        prob = F.softmax(logits, dim=1)
        correct_prob = prob * onehot_labels

        correct_prob = torch.sum(correct_prob, dim=1, keepdim=True)
        other_prob = prob * (1.0 - onehot_labels)

        if self.top_k > 1:
            topk_prob, _ = other_prob.topk(self.top_k, dim=1)
        else:
            topk_prob, _ = other_prob.max(dim=1, keepdim=True)

        diff_prob = correct_prob - topk_prob

        loss = torch.empty(0, device=logits.device)
        for feature_map in feature_maps:
            diff_grad = torch.stack([_get_grad(diff_prob[:, i], feature_map) for i in range(self.top_k)],
                                    dim=1)
            diff_gradnorm = torch.norm(diff_grad, p=self.dual_norm, dim=2)

            if self.use_approximation:
                diff_gradnorm.detach_()

            dist_to_boundary = diff_prob / (diff_gradnorm + self.eps)

            if self.loss_type == "worst_top_k":
                dist_to_boundary, _ = dist_to_boundary.min(dim=1)
            elif self.loss_type == "avg_top_k":
                dist_to_boundary = dist_to_boundary.mean(dim=1)

            loss_layer = _max_with_relu(dist_to_boundary, self.dist_lower)
            loss_layer = _max_with_relu(0, self.dist_upper - loss_layer) - self.dist_upper
            loss = torch.cat([loss, loss_layer])
        return loss.mean()

