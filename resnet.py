"""Create classifier architecture
Base model is ResNet
Returns:
    [nn.Module] -- RenNet classifier
"""
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic block for resnet under 50 layers
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, batch_size):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            self._norm_layer(batch_size, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels * BasicBlock.expansion,
                kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            self._norm_layer(batch_size, out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
                self._norm_layer(batch_size, out_channels * BasicBlock.expansion)
            )

    def _norm_layer(self, batch_size, out_channels):
        if batch_size <= 32:
            return nn.GroupNorm(32, out_channels)
        else:
            return nn.BatchNorm2d(out_channels)

    def forward(self, x):
        _out = self.residual_function(x) + self.shortcut(x)
        out = nn.ReLU(True)(_out)
        return out


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * BottleNeck.expansion,
                    stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        _out = self.residual_function(x) + self.shortcut(x)
        out = nn.ReLU(inplace=True)(_out)
        return out


class ResNet(nn.Module):
    """Build ResNet
    """
    def __init__(self, block, num_block, num_classes, input_channels, batch_size):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            self._norm_layer(batch_size, 64),
            nn.ReLU(inplace=True),
        )
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, batch_size)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, batch_size)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, batch_size)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, batch_size)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, batch_size):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, batch_size))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _norm_layer(self, batch_size, out_channels):
        if batch_size <= 32:
            return nn.GroupNorm(32, out_channels)
        else:
            return nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        features = output.view(output.size(0), -1)
        output = self.fc(features)

        return output, features


def resnet18(num_classes, channels, batch_size):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, channels, batch_size)

def resnet34(num_classes, channels, batch_size):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, channels, batch_size)

def resnet50(num_classes, channels, batch_size):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, channels, batch_size)

def resnet101(num_classes, channels, batch_size):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, channels, batch_size)

def resnet152(num_classes, channels, batch_size):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes, channels, batch_size)

