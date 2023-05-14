
import torch
import torch.nn as nn
from torch.autograd import Variable


NC=3

class vgg8_embed(nn.Module):
    def __init__(self, dim_embed=128):
        super(vgg8_embed, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'])
        
        self.x2h_res = nn.Sequential(
            nn.Linear(4608, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, dim_embed),
            nn.BatchNorm1d(dim_embed),
            nn.ReLU(),
        )

        self.h2y = nn.Sequential(
            nn.Linear(dim_embed, 1),
            nn.ReLU()
        )

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

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        features = self.x2h_res(features)
        out = self.h2y(features)
        return out, features


#------------------------------------------------------------------------------
# map labels to the embedding space
class model_y2h(nn.Module):
    def __init__(self, dim_embed=128):
        super(model_y2h, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.ReLU()
        )

    def forward(self, y):
        y = y.view(-1, 1) + 1e-8
        # y = torch.exp(y.view(-1, 1))
        return self.main(y)




if __name__ == "__main__":

    net = vgg8_embed(dim_embed=128).cuda()
    x = torch.randn(4,3,224,224).cuda()
    out, features = net(x)
    print(out.size())
    print(features.size())

    net_y2h = model_y2h()