import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class Custom_ResNet18(ResNet):
    def __init__(self):
        super(Custom_ResNet18, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=10
        ) # Based on ResNet18
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        backbone = Custom_ResNet18()
        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )
        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        features = out.view(out.size(0), -1)
        logit = self.fc(features)

        return logit, features


class Rep(nn.Module):
    def __init__(self, n_classes=10):
        super(Rep, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5),
            nn.PReLU(),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.Conv2d(64, 64, 5),
            nn.PReLU(),
            nn.Conv2d(64, 128, 5),
            nn.PReLU(),
            nn.Conv2d(128, 128, 5),
            nn.PReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, n_classes)
        )
        self.avg_pool = nn.AvgPool2d(4)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)
        out = self.fc(features)

        return out, features


class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        features = x.view(x.size(0), -1)
        logits = self.fc(features)
        return logits, features


def smallnet(model_type):
    if model_type == '18':
        return ResNet18()
    elif model_type == 'lenet':
        return LeNet5()
    else:
        return Rep()
