'''

Density Ration Approximation via Multilayer Perceptron

Multilayer Perceptron : trained to model density ratio in a feature space; based on "Rectified Linear Units Improve Restricted Boltzmann Machines"

Its input is the output of a pretrained Deep CNN, say ResNet-34

'''

import torch
import torch.nn as nn


IMG_SIZE=224


class ConditionalNorm2d(nn.Module):
    def __init__(self, num_features, dim_cond, dim_group=None):
        super().__init__()
        self.num_features = num_features
        # self.norm = nn.BatchNorm2d(num_features, affine=False)
        self.norm = nn.GroupNorm(dim_group, num_features, affine=False)

        self.embed_gamma = nn.Linear(dim_cond, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_cond, num_features, bias=False)

    def forward(self, x, y):
        out = self.norm(x)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out



class cDR_CNN(nn.Module):
    def __init__(self, img_size=IMG_SIZE, dim_cond=128):
        super(cDR_CNN, self).__init__()
        self.img_size = img_size
        self.dim_cond = dim_cond

        self.conv1 = nn.Conv2d(12, 128, kernel_size=4, stride=2, padding=1) 
        self.norm1 = ConditionalNorm2d(128, dim_cond, dim_group=8)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) 
        self.norm2 = ConditionalNorm2d(256, dim_cond, dim_group=8)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) 
        self.norm3 = ConditionalNorm2d(512, dim_cond, dim_group=16)

        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1) 
        self.norm4 = ConditionalNorm2d(1024, dim_cond, dim_group=16)

        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=0) 
        self.norm5 = ConditionalNorm2d(1024, dim_cond, dim_group=16)
        
        self.relu = nn.ReLU()

        self.final = nn.Sequential(
            nn.Linear(4096, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )
    
    def forward(self, x, y):
        x = x.view(x.size(0), 12, self.img_size//2, self.img_size//2)
        y = y.view(y.size(0), -1)

        ## layer 1
        out = self.conv1(x)
        out = self.norm1(out, y)
        out = self.relu(out)

        ## layer 2
        out = self.conv2(out)
        out = self.norm2(out, y)
        out = self.relu(out)

        ## layer 3
        out = self.conv3(out)
        out = self.norm3(out, y)
        out = self.relu(out)

        ## layer 4
        out = self.conv4(out)
        out = self.norm4(out, y)
        out = self.relu(out)

        ## layer 5
        out = self.conv5(out)
        out = self.norm5(out, y)
        out = self.relu(out)

        ##final
        out = out.view(out.size(0),-1)
        out = self.final(out)

        return out


if __name__ == "__main__":
    init_in_dim = 2
    net = cDR_CNN(img_size=224, dim_cond=128).cuda()
    x = torch.randn((10,224**2*3)).cuda()
    labels = torch.randn((10, 128)).cuda()
    out = net(x, labels)
    print(out.size())

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(net))
