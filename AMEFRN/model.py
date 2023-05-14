import torch
import torch.nn as nn
from torchvision import models as torch_models


NC=3

class vgg(nn.Module):
    def __init__(self, vgg_name, pretrained=True):
        super(vgg, self).__init__()
        command_exec = "self.conv = torch_models.{}(pretrained={},progress=True)".format(vgg_name, pretrained)
        exec(command_exec)
        
        self.conv = nn.Sequential(*list(self.conv.children())[:-2])
        
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
                
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



if __name__ == "__main__":
    net = vgg("vgg16_bn").cuda()
    # net = nn.DataParallel(net)
    x = torch.randn(4,3,224,224).cuda()
    print(net(x).size())