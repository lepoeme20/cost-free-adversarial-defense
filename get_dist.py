# -*- coding: utf-8 -*-
import os
import torch
from utils import (
    network_initialization,
    get_dataloader,
    get_m_s,
    norm,
    get_center,
    set_seed
)
import config


class Test:
    def __init__(self, args, phase):
        set_seed(args.seed)
        # self.args = args
        self.init_model = network_initialization(args)
        _, _, self.data_loader = get_dataloader(args)
        self.m, self.s = get_m_s(args)
        self.model_path = os.path.join(
            args.save_path,
            args.dataset,
            f'{phase}_model_110.pt'
            # 'intra_only_110.pt'
        )
        self.phase = phase

    def load_model(self, model, load_path):
        checkpoint = torch.load(load_path)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        return model

    def get_dist(self):
        model = self.load_model(self.init_model, self.model_path)
        center = get_center(
            model, self.data_loader, args.num_class, args.device, self.m, self.s
        )
        model.eval()
        dist = None
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(self.data_loader):
                imgs = norm(inputs, self.m, self.s)
                _, features = model(imgs)
                batch_dist = torch.zeros(args.num_class)
                for label in torch.unique(labels):
                    label_idx = torch.where(labels==label)[0]
                    class_features = features[label_idx]
                    batch_mean_dist = torch.mean(torch.cdist(center[label].unsqueeze(0), class_features))
                    batch_dist[label] = batch_mean_dist
                if dist is None:
                    dist = batch_dist.unsqueeze(0)
                else:
                    dist = torch.cat((dist, batch_dist.unsqueeze(0)))
        mean_dist = torch.mean(dist, 0)
        print(f"[{self.phase}] {args.dataset}: {mean_dist}")

        file_name = f'./{self.phase}_{args.dataset}_{str(torch.mean(mean_dist).item()).replace(".", "_")}.txt'
        with open(file_name, 'w') as f:
            f.write(str(mean_dist.tolist()))

if __name__ == "__main__":
    # intra, restricted
    for phase in ['only', 'restricted', 'intra']:
        for data in ['cifar10', 'cifar100', 'svhn']:
            args = config.get_config()
            args.dataset = data
            if data == 'cifar100':
                args.num_class = 100
            module = Test(args, phase)
            module.get_dist()
