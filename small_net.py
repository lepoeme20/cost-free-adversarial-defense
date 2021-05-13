import torch
import torch.nn as nn

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
        self.avg_pool = nn.AvgPool2d(8)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)
        out = self.fc(features)

        return out, features

class LeNet5_(nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits, features


def smallnet():
    return LeNet5()
