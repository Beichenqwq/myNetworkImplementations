import sys

import torch
from torch import nn

class Block(nn.Module):
    expansion = 4
    def __init__(self, in_channel, mid_channel, downsample=None, fstride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        #
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, 3, fstride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        #
        self.conv3 = nn.Conv2d(mid_channel, mid_channel*self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        rx = x

        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.downsample:
            rx = self.downsample(x)

        y += rx
        y = self.relu(y)
        return y

class Resnet50(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000):
        super(Resnet50, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self.stage_constructor(block, 64, block_nums[0])
        self.stage2 = self.stage_constructor(block, 128, block_nums[1], fstride=2)
        self.stage3 = self.stage_constructor(block, 256, block_nums[2], fstride=2)
        self.stage4 = self.stage_constructor(block, 512, block_nums[3], fstride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        print(x.shape)
        x = self.stage1(x)
        print(x.shape)
        x = self.stage2(x)
        print(x.shape)
        x = self.stage3(x)
        print(x.shape)
        x = self.stage4(x)
        print(x.shape)
        # sys.exit()
        x = self.avgpool(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)

        return x

    def stage_constructor(self, block, mid_channel, block_num, fstride=1): # fstride is 1 when size doesnot change, 2 when it halves.
        downsample = None
        if fstride != 1 or self.in_channel != mid_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, mid_channel * block.expansion, kernel_size=1, stride=fstride, bias=False),
                nn.BatchNorm2d(mid_channel * block.expansion))

        stage = []
        stage.append(block(self.in_channel, mid_channel, downsample=downsample, fstride=fstride))
        self.in_channel = mid_channel * block.expansion

        for _ in range(1, block_num):
            stage.append(block(self.in_channel, mid_channel))

        return nn.Sequential(*stage)

if __name__ == "__main__":
    net = Resnet50(Block, [3, 4, 6, 3])
    x = torch.normal(0, 1, [1, 3, 224, 224])
    y = net(x)
    print(y.shape)
