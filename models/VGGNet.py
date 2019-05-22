# VGGNet.py
import torch.nn as nn


def make_layers(cfg, dtype, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers).type(dtype)

class VGGNet11(nn.Module):
    def __init__(self, num_classes, dtype):
        super(VGGNet11, self).__init__()
        self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = make_layers(self.cfg, dtype)
        self.classifier = nn.Linear(512, num_classes).type(dtype)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGGNet11_bn(nn.Module):
    def __init__(self, num_classes, dtype):
        super(VGGNet11_bn, self).__init__()
        self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = make_layers(self.cfg, dtype, True)
        self.classifier = nn.Linear(512, num_classes).type(dtype)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x