from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['resnet']


config = {
    "resnet8": 576,
    "resnet14": 576,
    "resnet20": 576,
    "resnet32": 576,
    "resnet44": 576,
    "resnet56": 576,
    "resnet110": 576,

    "resnet8x4": 2304,
    "resnet32x4": 2304,
}



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):

    def __init__(self, model_name, depth, num_filters, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False) #h=h
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n, stride=2)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)

        self.maxpool = nn.MaxPool2d((2,2))
        self.avgpool = nn.AvgPool2d((2,2))

        feature_dim = config[model_name]
        self.fc = nn.Sequential(
                nn.Linear(feature_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                    
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                
                nn.Linear(512, 1),
                nn.ReLU(),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def forward(self, x):

        feat = self.relu(self.bn1(self.conv1(x)))
        feat, _ = self.layer1(feat)  # 112
        feat = self.maxpool(feat) #56
        feat, _ = self.layer2(feat)  # 28
        feat = self.maxpool(feat) #14
        feat, _ = self.layer3(feat)  # 7

        feat = self.avgpool(feat) #256 x 3 x 3

        out = feat.view(feat.size(0), -1)
        out = self.fc(out)

        return out, feat


def resnet8(**kwargs):
    return ResNet("resnet8", 8, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet14(**kwargs):
    return ResNet("resnet14", 14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20(**kwargs):
    return ResNet("resnet20", 20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet32(**kwargs):
    return ResNet("resnet32", 32, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet44(**kwargs):
    return ResNet("resnet44", 44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet56(**kwargs):
    return ResNet("resnet56", 56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet110(**kwargs):
    return ResNet("resnet110", 110, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet8x4(**kwargs):
    return ResNet("resnet8x4", 8, [32, 64, 128, 256], 'basicblock', **kwargs)


def resnet32x4(**kwargs):
    return ResNet("resnet32x4", 32, [32, 64, 128, 256], 'basicblock', **kwargs)


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 224, 224)
    net = resnet8()
    out,feat = net(x)
    print(out.size())
    print(feat.size())
