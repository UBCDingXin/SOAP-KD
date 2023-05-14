'''

SAGAN arch

Adapted from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py

'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))


def sndeconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out




'''

Generator

'''


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + gamma*out + beta
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_embed):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels):
        x0 = x

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class SAGAN_Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim=256, nc=3, gene_ch=64, dim_embed=128):
        super(SAGAN_Generator, self).__init__()

        self.z_dim = z_dim
        self.gene_ch = gene_ch
        self.snlinear0 = snlinear(in_features=z_dim, out_features=gene_ch*16*7*7)

        self.block1 = GenBlock(gene_ch*16, gene_ch*8, dim_embed)
        self.block2 = GenBlock(gene_ch*8, gene_ch*8, dim_embed)
        self.block3 = GenBlock(gene_ch*8, gene_ch*4, dim_embed)
        self.self_attn = Self_Attn(gene_ch*4)
        self.block4 = GenBlock(gene_ch*4, gene_ch*2, dim_embed)
        self.block5 = GenBlock(gene_ch*2, gene_ch, dim_embed)
        self.bn = nn.BatchNorm2d(gene_ch, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=gene_ch, out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # n x z_dim
        out = self.snlinear0(z)            # 4*4
        out = out.view(-1, self.gene_ch*16, 7,7) # 7x7
        out = self.block1(out, labels)    # 14 x 14
        out = self.block2(out, labels)    # 28 x 28
        out = self.block3(out, labels)    # 56 x 56
        out = self.self_attn(out)         # 56 x 56
        out = self.block4(out, labels)    # 112 x 112
        out = self.block5(out, labels)    # 224 x 224
        out = self.bn(out)
        out = self.relu(out)
        out = self.snconv2d1(out)
        out = self.tanh(out)
        return out



'''

Discriminator

'''


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class SAGAN_Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, nc=3, disc_ch=64, dim_embed=128):
        super(SAGAN_Discriminator, self).__init__()
        self.disc_ch = disc_ch
        self.opt_block1 = DiscOptBlock(nc, disc_ch)
        self.block1 = DiscBlock(disc_ch, disc_ch*2)
        self.self_attn = Self_Attn(disc_ch*2)
        self.block2 = DiscBlock(disc_ch*2, disc_ch*4)
        self.block3 = DiscBlock(disc_ch*4, disc_ch*8)
        self.block4 = DiscBlock(disc_ch*8, disc_ch*16)
        self.relu = nn.ReLU(inplace=True)
        
        self.snlinear1 = snlinear(in_features=disc_ch*16*7*7, out_features=1)
        self.sn_embedding1 = snlinear(dim_embed, disc_ch*16*7*7, bias=False)

        # Weight init
        self.apply(init_weights)
        xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        # 224x224
        out = self.opt_block1(x)  # 112 x 112
        out = self.block1(out)    # 56 x 56
        out = self.self_attn(out) # 56 x 56
        out = self.block2(out)    # 28 x 28
        out = self.block3(out)    # 14 x 14
        out = self.block4(out, downsample=True)    # 7 x 7
        out = self.relu(out)              # n x disc_ch*16 x 7 x 7
        out = out.view(-1, self.disc_ch*16*7*7) # n x (disc_ch*16*4*4)
        output1 = torch.squeeze(self.snlinear1(out)) # n
        # Projection
        h_labels = self.sn_embedding1(labels)   # n x disc_ch*16
        proj = torch.mul(out, h_labels)          # n x disc_ch*16
        output2 = torch.sum(proj, dim=[1])      # n
        # Out
        output = output1 + output2              # n
        return output


if __name__ == "__main__":
    netG = SAGAN_Generator(z_dim=256, gene_ch=64, dim_embed=128).cuda()
    netD = SAGAN_Discriminator(disc_ch=64, dim_embed=128).cuda()

    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    N=8
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

