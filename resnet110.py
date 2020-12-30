"""
Created on Tue Apr  2 14:21:30 2019

@author: aamir-mustafa
"""
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  # (conv-bn-relu) x 3 times
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity   # in our case is none
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth=110, block_name='BasicBlock', num_classes=100):
        super(ResNet, self).__init__()
        if block_name.lower() == 'basicblock':
            self.inplanes = 64
            out_ch = [64, 128, 256]
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            self.inplanes = 16
            out_ch = [16, 32, 64]
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')
        self.conv1 = nn.Conv2d(3, out_ch[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, out_ch[0], n)
        self.layer2 = self._make_layer(block, out_ch[1], n, stride=2)
        self.layer3 = self._make_layer(block, out_ch[2], n, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(256, num_classes)
        # baseline
        # self.fc = nn.Linear(64 * block.expansion, 1024)
        # self.fcf = nn.Linear(1024, num_classes)

        # self.maxpool2 = nn.MaxPool2d(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # m =  self.maxpool2(x)
        # m = m.view(m.size(0), -1) # 128 dimensional

        x = self.layer3(x)

        x = self.avgpool(x)
        z = x.view(x.size(0), -1) # 256 dimensional
        x = self.fc(z)            # 1024 dimensional
        # y = self.fcf(x)           # num_classes dimensional

        # return y, m, z, x
        return x, x, z, z

def resnet(num_classes, block='BasicBlock'):
    """
    Constructs a ResNet model.
    """
    return ResNet(num_classes=num_classes, block_name=block)

