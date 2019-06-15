import math
import torch.nn as nn
import torch.nn.functional as F


def round_fn(orig, multiplier):
    if not multiplier:
        return orig

    return int(math.ceil(multiplier * orig))


def get_activation_fn(activation):
    if activation == "swish":
        return Swish

    elif activation == "relu":
        return nn.ReLU

    else:
        raise Exception('Unkown activation %s' % activation)


class Swish(nn.Module):
    """ Swish activation function, s(x) = x * sigmoid(x) """

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class ConvBlock(nn.Module):
    """ Conv + BatchNorm + Activation """

    def __init__(self, in_channel, out_channel, kernel_size,
                 padding=0, stride=1, activation="swish"):
        super().__init__()
        self.fw = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size,
                          padding=padding, stride=stride),
                nn.BatchNorm2d(out_channel),
                get_activation_fn(activation)())

    def forward(self, x):
        return self.fw(x)


class DepthwiseConvBlock(nn.Module):
    """ DepthwiseConv2D + BatchNorm + Activation """

    def __init__(self, in_channel, kernel_size,
                 padding=0, stride=1, activation="swish"):
        super().__init__()
        self.fw = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size,
                          padding=padding, stride=stride, groups=in_channel),
                nn.BatchNorm2d(in_channel),
                get_activation_fn(activation)())

    def forward(self, x):
        return self.fw(x)


class MBConv(nn.Module):
    """ Inverted residual block """

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, expand_ratio=1, activation="swish"):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.expand_ratio = expand_ratio
        self.stride = stride

        if expand_ratio != 1:
            self.expand = ConvBlock(in_channel, in_channel*expand_ratio, 1,
                                    activation=activation)

        self.dw_conv = DepthwiseConvBlock(in_channel*expand_ratio, kernel_size,
                                          padding=(kernel_size-1)//2,
                                          stride=stride, activation=activation)

        self.pw_conv = ConvBlock(in_channel*expand_ratio, out_channel, 1,
                                 activation=activation)

    def forward(self, inputs):
        if self.expand_ratio != 1:
            x = self.expand(inputs)
        else:
            x = inputs

        x = self.dw_conv(x)
        x = self.pw_conv(x)

        if self.in_channel * self.expand_ratio == self.out_channel and \
                self.stride == 1:
            x = x + inputs

        return x


class Net(nn.Module):
    """ EfficientNet """

    def __init__(self, pi=0, activation="swish", num_classes=1000):
        super(Net, self).__init__()

        self.d = 1.2 ** pi
        self.w = 1.1 ** pi
        self.r = 1.15 ** pi
        self.img_size = (round_fn(224, self.r), round_fn(224, self.r))

        self.stage1 = ConvBlock(3, round_fn(32, self.w),
                                kernel_size=3, padding=1, stride=2, activation=activation)

        self.stage2 = self.make_layers(round_fn(32, self.w), round_fn(16, self.w),
                                       depth=round_fn(1, self.d), kernel_size=3,
                                       half_resolution=False, expand_ratio=1, activation=activation)

        self.stage3 = self.make_layers(round_fn(16, self.w), round_fn(24, self.w),
                                       depth=round_fn(2, self.d), kernel_size=3,
                                       half_resolution=True, expand_ratio=6, activation=activation)

        self.stage4 = self.make_layers(round_fn(24, self.w), round_fn(40, self.w),
                                       depth=round_fn(2, self.d), kernel_size=5,
                                       half_resolution=True, expand_ratio=6, activation=activation)

        self.stage5 = self.make_layers(round_fn(40, self.w), round_fn(80, self.w),
                                       depth=round_fn(3, self.d), kernel_size=3,
                                       half_resolution=True, expand_ratio=6, activation=activation)

        self.stage6 = self.make_layers(round_fn(80, self.w), round_fn(112, self.w),
                                       depth=round_fn(3, self.d), kernel_size=5,
                                       half_resolution=False, expand_ratio=6, activation=activation)

        self.stage7 = self.make_layers(round_fn(112, self.w), round_fn(192, self.w),
                                       depth=round_fn(4, self.d), kernel_size=7,
                                       half_resolution=True, expand_ratio=6, activation=activation)

        self.stage8 = self.make_layers(round_fn(192, self.w), round_fn(320, self.w),
                                       depth=round_fn(1, self.d), kernel_size=3,
                                       half_resolution=False, expand_ratio=6, activation=activation)

        self.stage9 = ConvBlock(round_fn(320, self.w), round_fn(1280, self.w),
                                kernel_size=1, activation=activation)

        self.fc = nn.Linear(round_fn(7*7*1280, self.w), num_classes)

    def make_layers(self, in_channel, out_channel, depth, kernel_size,
                    half_resolution=False, expand_ratio=1, activation="swish"):
        blocks = []
        for i in range(depth):
            stride = 2 if half_resolution and i==0 else 1
            blocks.append(
                    MBConv(in_channel, out_channel, kernel_size,
                           stride=stride, expand_ratio=expand_ratio, activation=activation))
            in_channel = out_channel

        return nn.Sequential(*blocks)

    def forward(self, x):
        assert x.size()[-2:] == self.img_size, \
                'Image size must be %r, but %r given' % (self.img_size, x.size()[-2])

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x, x
