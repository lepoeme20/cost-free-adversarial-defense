# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import (
    network_initialization,
    get_dataloader,
    get_m_s,
    norm,
    get_center,
    set_seed
)
import config
from attack_methods import pgd, fgsm, cw, bim


class Test:
    def __init__(self, args):
        set_seed(args.seed)
        # self.args = args
        self.model = network_initialization(args)
        model_path = os.path.join(
            args.save_path,
            args.dataset,
        )
        if not args.adv_train:
            self.model_path = os.path.join(
                model_path, f"{args.test_model}_model_{args.model}.pt"
            )
        else:
            self.model_path = os.path.join(model_path, f"{args.test_model}_model_adv_train.pt")

    def load_model(self, model, load_path):
        checkpoint = torch.load(load_path)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint

    def attack(self, target_cls, dataloader):
        attack_module = globals()[args.attack_name.lower()]
        attack_func = getattr(attack_module, args.attack_name)
        attacker = attack_func(target_cls, args)
        save_path = os.path.join("Adv_examples", args.dataset.lower())
        attacker.inference(
            args,
            data_loader=dataloader,
            save_path=save_path,
            file_name=args.attack_name + ".pt",
        )

    def testing(self):
        model, checkpoint = self.load_model(self.model, self.model_path)

        args.batch_size = args.test_batch_size
        train_loader, _, tst_loader = get_dataloader(args)
        m, s = get_m_s(args)

        # 정상 데이터에 대한 모델 성능 확인
        if args.attack_name.lower() == "clean":
            correct = 0
            accumulated_num = 0.0
            total_num = len(tst_loader)

            for step, (inputs, labels) in enumerate(tst_loader, 0):
                model.eval()
                if inputs.size(1) == 1:
                    inputs = inputs.expand(inputs.size(0), 3, inputs.size(2), inputs.size(3))
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                accumulated_num += labels.size(0)
                inputs = norm(inputs, m, s)

                with torch.no_grad():
                    outputs, features = model(inputs)
                    # dist = torch.cdist(features, center, p=2)
                    # print(dist)
                    _, predicted = torch.max(outputs, 1)
                    correct += predicted.eq(labels).sum().item()

                    acc = 100 * correct / accumulated_num

                    print(
                        "Progress : {:.2f}% / Accuracy : {:.2f}%".format(
                            (step + 1) / total_num * 100, acc
                        ),
                        end="\r",
                    )

            print(
                "Progress : {:.2f}% / Accuracy : {:.2f}%".format(
                    (step + 1) / total_num * 100, acc
                )
            )
        # attack시 방어 성능 확인
        else:
            self.attack(model, tst_loader)


if __name__ == "__main__":
    args = config.get_config()
    tester = Test(args)

    tester.testing()
