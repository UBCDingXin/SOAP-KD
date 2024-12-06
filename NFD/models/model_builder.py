import torch
import torch.nn as nn
from torchvision import models as torch_models


NC=3

feature_dims = {
    "vgg11": 4608,
    "vgg13": 4608,
    "vgg16": 4608,
    "vgg19": 4608, ## 512, 3, 3

    "vgg11_bn": 4608,
    "vgg13_bn": 4608,
    "vgg16_bn": 4608,
    "vgg19_bn": 4608,

    "resnet18": 4608,
    "resnet34": 4608,
    "resnet50": 18432, 
    "resnet101": 18432, ##2048, 3, 3
 
    "resnext50_32x4d": 18432,
    "resnext101_32x8d": 18432,

    "wide_resnet50_2": 18432,
    "wide_resnet101_2": 18432,

    "shufflenet_v2_x0_5": 1728, ##192, 3, 3
    "shufflenet_v2_x1_0": 4176, ##464, 3, 3

    "densenet121": 9216,
    "densenet169": 14976,
    "densenet201": 17280,
    "densenet161": 19872, ##2208, 3, 3

    "mobilenet_v2": 11520, ##1280, 3, 3
    
    "efficientnet_b0": 5120,
}




def load_fn(model_name, pretrained=True):
    if model_name=="vgg11":
        return torch_models.vgg11(pretrained=pretrained,progress=True)
    elif model_name=="vgg13":
        return torch_models.vgg13(pretrained=pretrained,progress=True)
    elif model_name=="vgg16":
        return torch_models.vgg16(pretrained=pretrained,progress=True)
    elif model_name=="vgg19":
        return torch_models.vgg19(pretrained=pretrained,progress=True)
    
    elif model_name=="vgg11_bn":
        return torch_models.vgg11_bn(pretrained=pretrained,progress=True)
    elif model_name=="vgg13_bn":
        return torch_models.vgg13_bn(pretrained=pretrained,progress=True)
    elif model_name=="vgg16_bn":
        return torch_models.vgg16_bn(pretrained=pretrained,progress=True)
    elif model_name=="vgg19_bn":
        return torch_models.vgg19_bn(pretrained=pretrained,progress=True)
    
    elif model_name=="resnet18":
        return torch_models.resnet18(pretrained=pretrained,progress=True)
    elif model_name=="resnet34":
        return torch_models.resnet34(pretrained=pretrained,progress=True)
    elif model_name=="resnet50":
        return torch_models.resnet50(pretrained=pretrained,progress=True)
    elif model_name=="resnet101":
        return torch_models.resnet101(pretrained=pretrained,progress=True)
    
    elif model_name=="resnext50_32x4d":
        return torch_models.resnext50_32x4d(pretrained=pretrained,progress=True)
    elif model_name=="resnext101_32x8d":
        return torch_models.resnext101_32x8d(pretrained=pretrained,progress=True)
    
    elif model_name=="wide_resnet50_2":
        return torch_models.wide_resnet50_2(pretrained=pretrained,progress=True)
    elif model_name=="wide_resnet101_2":
        return torch_models.wide_resnet101_2(pretrained=pretrained,progress=True)
    
    elif model_name=="shufflenet_v2_x0_5":
        return torch_models.shufflenet_v2_x0_5(pretrained=pretrained,progress=True)
    elif model_name=="shufflenet_v2_x1_0":
        return torch_models.shufflenet_v2_x1_0(pretrained=pretrained,progress=True)
    
    elif model_name=="densenet121":
        return torch_models.densenet121(pretrained=pretrained,progress=True)
    elif model_name=="densenet169":
        return torch_models.densenet169(pretrained=pretrained,progress=True)
    elif model_name=="densenet201":
        return torch_models.densenet201(pretrained=pretrained,progress=True)
    elif model_name=="densenet161":
        return torch_models.densenet161(pretrained=pretrained,progress=True)
    
    elif model_name=="mobilenet_v2":
        return torch_models.mobilenet_v2(pretrained=pretrained,progress=True)
    
    elif model_name=="efficientnet_b0":
        return torch_models.efficientnet_b0(pretrained=pretrained,progress=True)




class model_builder(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(model_builder, self).__init__()

        if model_name in feature_dims:
            feature_dim = feature_dims[model_name]
        else:
            raise Exception("Not supported network...")

        # command_exec = "self.conv = torch_models.{}(pretrained={},progress=True)".format(model_name, pretrained)
        # exec(command_exec)
       
        conv_blocks = load_fn(model_name=model_name, pretrained=pretrained)
        if ("mobilenet" in model_name) or ("densenet" in model_name):
            conv_blocks = list(conv_blocks.children())[:-1]
        else:
            conv_blocks = list(conv_blocks.children())[:-2]
        conv_blocks += [nn.AvgPool2d((2,2))]
        self.conv = nn.Sequential(*conv_blocks)

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
        feat = self.conv(x)
        out = feat.view(feat.size(0), -1)
        out = self.fc(out)
        return out, feat



if __name__ == "__main__":
    net = model_builder(model_name="vgg19").cuda()
    net = nn.DataParallel(net)
    x = torch.randn(4,3,224,224).cuda()
    out, feat = net(x)
    print(out.size())
    print(feat.size())

    # net = model_builder(model_name="vgg19").cuda()
    # filename_ckpt = "D:/BaiduSyncdisk/Baidu_WD/remote_sensing/SOAP-KD/SOAP-KD/output/CNN/vanilla/ckpt_vgg19_epoch_200_pretrain_False_last.pth"
    # checkpoint = torch.load(filename_ckpt)
    # net.load_state_dict(checkpoint['net_state_dict'])
    # x = torch.randn(4,3,224,224).cuda()
    # out, feat = net(x)
    # print(out.size())
    # print(feat.size())
