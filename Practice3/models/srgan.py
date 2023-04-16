from math import log2

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, shortcut=True, e=1):
        super().__init__()
        _out_channels = int(out_channels * e)

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, _out_channels, 3, 1, 1),
            nn.BatchNorm2d(_out_channels),
            nn.PReLU(),

            nn.Conv2d(_out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.res_block(x) if self.add else self.res_block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, scale_resize):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_resize ** 2), 3, 1, 1)
        self.upsample = nn.PixelShuffle(scale_resize)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.upsample(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, in_channels=3, scale_factor=4, nb=6):
        super().__init__()
        scale_resize = int(log2(scale_factor))  # number of upsample_block

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
            nn.PReLU(),
        )
        self.residuals = nn.Sequential(*[Bottleneck() for _ in range(nb)])  # nb: number of bottleneck
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.upsamples = nn.Sequential(
            Upsample(64, scale_resize),
            nn.Conv2d(64, 3, 9, 1, 4),
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.residuals(y1)
        y = self.upsamples(y1 + y2)

        return (torch.tanh(y) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.conv_net(x).view(x.size(0)))  # x.size(0): batch_size
