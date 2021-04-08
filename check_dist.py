import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import network_initialization, get_dataloader, get_m_s, norm
import config


def _compute_distance(features, labels):
    dist_mat = torch.cdist(features, features, p=2)

    dist = []
    for i in range(features.size(0)):
        value = dist_mat[i][torch.where(labels==labels[i])].mean()
        dist.append(value.item())
    mean_dist = np.mean(dist)

    return mean_dist


def check_distance(args, model_name):
    trn_loader, dev_loader, tst_loader = get_dataloader(args)
    model = network_initialization(args)
    m, s = get_m_s(args)

    root_path = os.path.join(args.save_path, args.dataset)
    model_path = os.path.join(root_path, f"{model_name}.pt")
    # proposed_path = os.path.join(root_path, f"proposed_model_test_inter.pt")

    model.module.load_state_dict(torch.load(model_path)["model_state_dict"])

    feature_dist = []
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(trn_loader):
            if labels.size(0) == args.batch_size:
                model.eval()

                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if inputs.size(1) == 1:
                    inputs = inputs.expand(inputs.size(0), 3, inputs.size(2), inputs.size(3))
                inputs = norm(inputs, m, s)

                _, features = model(inputs)
                dist = _compute_distance(features, labels)
                feature_dist.append(dist.item())

        mean_dist = np.mean(feature_dist)
        print(mean_dist)

if __name__=="__main__":
    args = config.get_config()
    args.data_root_path = '/repo/data'
    args.dataset = 'mnist'
    # model = 'pretrained_model_inter'
    model = 'ce_loss'
    args.num_class = 10


    check_distance(args, model)
