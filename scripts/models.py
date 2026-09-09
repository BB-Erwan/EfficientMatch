"""WideResNet, architecture standard FixMatch (Oliver et al. protocol)."""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_io = in_planes == out_planes and stride == 1
        self.shortcut = None if self.equal_io else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.equal_io else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=2, drop_rate=0.0):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, n_channels[0], 3, stride=1, padding=1, bias=False)
        self.block1 = self._make_block(n_channels[0], n_channels[1], n, stride=1, drop_rate=drop_rate)
        self.block2 = self._make_block(n_channels[1], n_channels[2], n, stride=2, drop_rate=drop_rate)
        self.block3 = self._make_block(n_channels[2], n_channels[3], n, stride=2, drop_rate=drop_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

    def _make_block(self, in_planes, out_planes, num_layers, stride, drop_rate):
        layers = [BasicBlock(in_planes, out_planes, stride, drop_rate)]
        for _ in range(1, num_layers):
            layers.append(BasicBlock(out_planes, out_planes, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(cfg, device):
    model = WideResNet(num_classes=cfg["num_classes"], depth=cfg["depth"], widen_factor=cfg["widen_factor"])
    return model.to(device)
