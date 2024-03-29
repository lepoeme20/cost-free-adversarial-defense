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
from attack_methods import pgd, fgsm, cw, bim, mim


class Test:
    def __init__(self, args):
        set_seed(args.seed)
        self.target = network_initialization(args)

        target_model_path = os.path.join(
            args.save_path,
            args.dataset,
            f"{args.test_model}_model_{args.model}.pt"
        )
        self.target.module.load_state_dict(
            torch.load(target_model_path)["model_state_dict"]
        )
        self.target.eval()

    def attack(self, tstloader):
        attack_module = globals()[args.attack_name.lower()]
        attack_func = getattr(attack_module, args.attack_name)
        attacker = attack_func(self.target, args)
        save_path = os.path.join("Adv_examples", args.dataset.lower())
        attacker.inference(
            args,
            data_loader=tstloader,
            save_path=save_path,
            file_name=args.attack_name + ".pt",
        )

    def testing(self):
        args.batch_size = args.test_batch_size
        _, _, tst_loader = get_dataloader(args)
        m, s = get_m_s(args)

        # 정상 데이터에 대한 모델 성능 확인
        if args.attack_name.lower() == "clean":
            correct = 0
            accumulated_num = 0.0
            total_num = len(tst_loader)

            for step, (inputs, labels) in enumerate(tst_loader, 0):
                if inputs.size(1) == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                accumulated_num += labels.size(0)
                inputs = norm(inputs, m, s)

                with torch.no_grad():
                    outputs, features = self.target(inputs)
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
            self.attack(tst_loader)


if __name__ == "__main__":
    args = config.get_config()
    tester = Test(args)

    tester.testing()
