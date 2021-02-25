import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU()
        )
        self.avg_pool = nn.AvgPool2d(6)
        self.fc_layer = nn.Sequential(
            nn.Linear(128, 10),
        )
    def forward(self,x):
        out = self.layer(x)
        out = self.avg_pool(out)
        features = out.view(out.size(0), -1)
        out = self.fc_layer(features)
        return out, features


class LeNet5(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)
        return logits, features


def smallnet():
    return LeNet5()
