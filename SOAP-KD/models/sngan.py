'''
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

chainer: https://github.com/pfnet-research/sngan_projection
'''

# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

# from spectral_normalization import SpectralNorm
import numpy as np
from torch.nn.utils import spectral_norm



class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, dim_embed, bias=True):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.condbn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.condbn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)

        # unconditional case
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=bias) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x, y):
        if y is not None:
            out = self.condbn1(x, y)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.conv1(out)
            out = self.condbn2(out, y)
            out = self.relu(out)
            out = self.conv2(out)
            out = out + self.bypass(x)
        else:
            out = self.model(x) + self.bypass(x)

        return out

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)



class SNGAN_Generator(nn.Module):
    def __init__(self, z_dim=256, nc=3, gene_ch=128, dim_embed=128):
        super(SNGAN_Generator, self).__init__()
        self.z_dim = z_dim
        self.dim_embed = dim_embed
        self.gene_ch = gene_ch

        self.dense = nn.Linear(self.z_dim, 7 * 7 * gene_ch*16, bias=True)
        self.final = nn.Conv2d(gene_ch, nc, 3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock0 = ResBlockGenerator(gene_ch*16, gene_ch*8, dim_embed=dim_embed) #7--->14
        self.genblock1 = ResBlockGenerator(gene_ch*8, gene_ch*8, dim_embed=dim_embed) #14--->28
        self.genblock2 = ResBlockGenerator(gene_ch*8, gene_ch*4, dim_embed=dim_embed) #28--->56
        self.genblock3 = ResBlockGenerator(gene_ch*4, gene_ch*2, dim_embed=dim_embed) #56--->112
        self.genblock4 = ResBlockGenerator(gene_ch*2, gene_ch, dim_embed=dim_embed) #112--->112

        self.final = nn.Sequential(
            nn.BatchNorm2d(gene_ch),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y): #y is embedded in the feature space
        z = z.view(z.size(0), z.size(1))
        out = self.dense(z)
        out = out.view(-1, self.gene_ch*16, 7, 7)

        out = self.genblock0(out, y)
        out = self.genblock1(out, y)
        out = self.genblock2(out, y)
        out = self.genblock3(out, y)
        out = self.genblock4(out, y)
        out = self.final(out)

        return out


class SNGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, disc_ch=128, dim_embed=128):
        super(SNGAN_Discriminator, self).__init__()
        self.dim_embed = dim_embed
        self.disc_ch = disc_ch

        self.discblock1 = nn.Sequential(
            FirstResBlockDiscriminator(nc, disc_ch, stride=2), #224--->112
            ResBlockDiscriminator(disc_ch, disc_ch*2, stride=2), #112--->56
            ResBlockDiscriminator(disc_ch*2, disc_ch*4, stride=2), #56--->28
        )
        self.discblock2 = ResBlockDiscriminator(disc_ch*4, disc_ch*8, stride=2) #28--->14
        self.discblock3 = nn.Sequential(
            ResBlockDiscriminator(disc_ch*8, disc_ch*16, stride=2), #14--->7;
            nn.ReLU(),
        )

        self.linear1 = nn.Linear(disc_ch*16*7*7, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        self.linear2 = nn.Linear(self.dim_embed, disc_ch*16*7*7, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)

    def forward(self, x, y):
        output = self.discblock1(x)
        output = self.discblock2(output)
        output = self.discblock3(output)

        output = output.view(-1, self.disc_ch*16*7*7)
        output_y = torch.sum(output*self.linear2(y), 1, keepdim=True)
        output = self.linear1(output) + output_y

        return output.view(-1, 1)



if __name__ == "__main__":
    netG = SNGAN_Generator(z_dim=256, gene_ch=64, dim_embed=128).cuda()
    netD = SNGAN_Discriminator(disc_ch=64, dim_embed=128).cuda()

    # netG = nn.DataParallel(netG)
    # netD = nn.DataParallel(netD)

    N=4
    z = torch.randn(N, 256).cuda()
    y = torch.randn(N, 128).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(x.size())
    print(o.size())


    def count_parameters(module, verbose=True):
        num_parameters = sum([p.data.nelement() for p in module.parameters()])
        if verbose:
            print('Number of parameters: {}'.format(num_parameters))
        return num_parameters
    
    count_parameters(netG)
    count_parameters(netD)