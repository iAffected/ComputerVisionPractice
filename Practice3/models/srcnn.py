from math import log2

import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 9, 1, 9 // 2)
        self.conv2 = nn.Conv2d(64, 32, 5, 1, 5 // 2)
        self.conv3 = nn.Conv2d(32, in_channels, 5, 1, 5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
