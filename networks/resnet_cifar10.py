import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(out_channel))

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel))

        if self.in_channel != self.out_channel or \
                self.stride != 1:
            self.down = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channel))

    def forward(self, b):
        t = self.conv1(b)
        t = self.relu(t)
        t = self.conv2(t)

        if self.in_channel != self.out_channel or \
                self.stride != 1:
            b = self.down(b)

        t += b
        t = self.relu(t)

        return t


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        scale = args.scale

        self.stem = nn.Sequential(
                nn.Conv2d(3, 16,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(*[
            ResidualBlock(16, 16, 1) for _ in range(2*scale)])

        self.layer2 = nn.Sequential(*[
            ResidualBlock(in_channel=(16 if i==0 else 32),
                          out_channel=32,
                          stride=(2 if i==0 else 1)) for i in range(2*scale)])

        self.layer3 = nn.Sequential(*[
            ResidualBlock(in_channel=(32 if i==0 else 64),
                          out_channel=64,
                          stride=(2 if i==0 else 1)) for i in range(2*scale)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(64, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s = self.stem(x)
        x = self.layer1(s)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x, s
