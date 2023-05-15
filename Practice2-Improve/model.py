import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


class Conv(nn.Module):  # convolution-normalization-activation
    default_norm_layer = nn.BatchNorm2d
    default_act = nn.ReLU()  # if inplace=True is required, do this in parameter 'act': act=nn.SiLU(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, bias=True, norm_layer=True, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), d, g, bias)
        self.bn = self.default_norm_layer(c2) if norm_layer is True \
            else norm_layer if isinstance(norm_layer, nn.Module) else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, g=1, e=0.5, bias=True, norm_layer=True, act=True, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, bias=bias, norm_layer=norm_layer, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, bias=bias, norm_layer=norm_layer, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            Conv(1, 32, 5, 1, 0),
            Conv(32, 32, 5, 1, 0),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            Conv(32, 64, 3, 1, 0),
            Conv(64, 64, 3, 1, 0),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Flatten(start_dim=1),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1),
        )

        initialize_weights(self)

    def forward(self, x):
        return self.backbone(x)
