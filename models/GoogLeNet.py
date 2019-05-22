# GoogLeNet.py
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduced, n3x3, n5x5_reduced, n5x5, pool_proj):
        super(Inception, self).__init__()
        self.sub_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1, stride=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True))
        
        self.sub_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduced, kernel_size=1, stride=1),
            nn.BatchNorm2d(n3x3_reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduced, n3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True))

        self.sub_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduced, stride=1, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduced, n5x5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True))
        
        self.sub_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, stride=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.sub_1x1(x), self.sub_3x3(x), self.sub_5x5(x), self.sub_pool(x)], dim=1)


class GoogleNet(nn.Module):
    def __init__(self, num_class, dtype):
        super(GoogleNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)).type(dtype)        
        self.features = nn.Sequential([
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(p=0.4),
        ]).type(dtype)
        self.linear =  nn.Linear(1024, num_class).type(dtype)
        
    def forward(self, x):
        output = self.pre_layer(x)
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output