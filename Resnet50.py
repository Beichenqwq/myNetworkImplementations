import sys

import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Block, self).__init__()
        self.downsample = False if in_channel==out_channel else True

        self.conv1 = nn.Conv2d(in_channel, mid_channel, 1, 1 if in_channel==mid_channel else 2, 0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(mid_channel, out_channel, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU()

        if self.downsample:
            self.rconv = nn.Conv2d(in_channel, out_channel, 1, 2, 0)
            self.rbn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        rx = x

        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(x)))
        y = self.relu3(self.bn3(self.conv3(x)))
        if self.downsample:
            rx = self.bn3(self.rconv(x))
        else:
            rx = x

        y += rx
        return rx

class Resnet50(nn.Module):
    def __init__(self, in_channel=64, num_classes=1000):
        super(Resnet50, self).__init__()
        self.conv1 = nn.Conv2d(3, in_channel, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self.stage_constructor(64, 256, 1) # TODO GENERALIZE
        self.stage2 = self.stage_constructor(256, 512, 2)
        self.stage3 = self.stage_constructor(512, 1024, 2)
        self.stage4 = self.stage_constructor(1024, 2048, 2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.stage1(x)
        x = self.stage2(x)
        #x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        print(x.shape)
        sys.exit()
        x = self.avgpool(x)

        x = self.fc(x)

    def stage_constructor(self, in_channel, out_channel, fstride):
        stage = [ConvBlock(in_channel, out_channel, fstride), IdentityBlock(out_channel), IdentityBlock(out_channel)]
        return nn.Sequential(*stage)

if __name__ == "__main__":
    net = nn.Sequential(Block(256, 64, 256), Block(256, 64, 256), Block(256, 64, 256))
    x = torch.normal(0, 1, [1, 256, 56, 56])
    y = net(x)
    print(y.shape)
