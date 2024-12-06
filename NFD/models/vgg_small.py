'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = {
    'VGG8':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
}

NC=3

class vgg(nn.Module):
    def __init__(self, vgg_name):
        super(vgg, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        feature_dim = 512*3*3
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

    def forward(self, x):
        feat = self.features(x) #512*3*3
        out = feat.view(feat.size(0), -1)
        out = self.fc(out)
        return out, feat

    def _make_layers(self, cfg):
        layers = []
        in_channels = NC
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

def vgg8():
    model = vgg('VGG8')
    return model


if __name__ == "__main__":
    net = vgg8().cuda()
    net = nn.DataParallel(net)
    x = torch.randn(4,3,224,224)
    out, feat = net(x)
    print(out.size())
    print(feat.size())