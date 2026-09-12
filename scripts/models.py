"""WideResNet, architecture standard FixMatch (Oliver et al. protocol).
DenseNet-BC pour CIFAR (Huang et al., 2016), adaptée pour images 32x32."""
import math
import torch
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


class BottleneckLayer(nn.Module):
    """Couche dense avec bottleneck : BN-ReLU-Conv1x1(4k) -> BN-ReLU-Conv3x3(k)"""

    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class TransitionLayer(nn.Module):
    """Compression entre les dense blocks : BN-ReLU-Conv1x1 -> AvgPool2x2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return F.avg_pool2d(out, 2)


class DenseNetCIFAR(nn.Module):
    def __init__(self, depth=100, growth_rate=12, reduction=0.5, num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth doit vérifier (depth-4) % 6 == 0 pour DenseNet-BC"
        n_layers_per_block = (depth - 4) // 6

        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)

        def make_dense_block(in_channels, n_layers):
            layers = []
            for i in range(n_layers):
                layers.append(BottleneckLayer(in_channels + i * growth_rate, growth_rate))
            return nn.Sequential(*layers), in_channels + n_layers * growth_rate

        self.block1, num_channels = make_dense_block(num_channels, n_layers_per_block)
        out_channels = int(math.floor(num_channels * reduction))
        self.trans1 = TransitionLayer(num_channels, out_channels)
        num_channels = out_channels

        self.block2, num_channels = make_dense_block(num_channels, n_layers_per_block)
        out_channels = int(math.floor(num_channels * reduction))
        self.trans2 = TransitionLayer(num_channels, out_channels)
        num_channels = out_channels

        self.block3, num_channels = make_dense_block(num_channels, n_layers_per_block)

        self.bn_final = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def densenet_bc_100_12(num_classes=100):
    """Config du papier original (L=100, k=12), ~0.8M paramètres."""
    return DenseNetCIFAR(depth=100, growth_rate=12, reduction=0.5, num_classes=num_classes)


def build_model(model_name, num_classes, widen_factor=2):
    """Factory commune aux scripts d'expérience : instancie le backbone choisi via --model."""
    if model_name == "wideresnet":
        return WideResNet(depth=28, widen_factor=widen_factor, num_classes=num_classes)
    elif model_name == "densenet":
        return densenet_bc_100_12(num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")
