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
from utils import network_initialization, get_dataloader
import config
from attack_methods import pgd, fgsm, cw, bim
from utils import get_m_s, norm


class Test:
    def __init__(self, args):
        self.args = args
        self.model = network_initialization(args)
        model_path = os.path.join(
            args.save_path,
            args.dataset,
        )
        self.model_path = os.path.join(model_path, f"{args.test_model}.pt")

    def load_model(self, model, load_path):
        checkpoint = torch.load(load_path)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        return model

    def attack(self, target_cls, dataloader):
        attack_module = globals()[self.args.attack_name.lower()]
        attack_func = getattr(attack_module, self.args.attack_name)
        attacker = attack_func(target_cls, self.args)
        save_path = os.path.join("Adv_examples", self.args.dataset.lower())
        attacker.inference(
            args,
            data_loader=dataloader,
            save_path=save_path,
            file_name=self.args.attack_name + ".pt",
        )

    def testing(self):
        # inter_model을 사용하고 싶은 경우 self.total_path > self.pre_path 변경
        model = self.load_model(self.model, self.model_path)

        args.batch_size = args.test_batch_size
        _, _, tst_loader = get_dataloader(self.args)
        m, s = get_m_s(args)

        # 정상 데이터에 대한 모델 성능 확인
        if self.args.attack_name.lower() == "clean":
            correct = 0
            accumulated_num = 0.0
            total_num = len(tst_loader)

            for step, (inputs, labels) in enumerate(tst_loader, 0):
                model.eval()
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                inputs = inputs.expand(inputs.size(0), 3, inputs.size(2), inputs.size(3))
                accumulated_num += labels.size(0)
                inputs = norm(inputs, m, s)

                with torch.no_grad():
                    outputs, _ = model(inputs)
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
