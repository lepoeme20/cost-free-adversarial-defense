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

def smallnet():
    return CNN()
