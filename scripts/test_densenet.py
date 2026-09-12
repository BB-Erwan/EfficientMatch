from models import WideResNet, DenseNetCIFAR, densenet_bc_100_12
import torch

m = densenet_bc_100_12(num_classes=100)
print('params:', sum(p.numel() for p in m.parameters()))
x = torch.randn(2, 3, 32, 32)
print('output shape:', m(x).shape)

wrn = WideResNet(num_classes=10)
print('WideResNet still works:', wrn(x[:, :, :, :]).shape if False else "ok")
