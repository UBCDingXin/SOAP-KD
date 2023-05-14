import torch
import torch.nn as nn
from torchvision import models as torch_models


NC=3
IMG_SIZE=224

feature_dims = {
    "vgg11": 512,
    "vgg13": 512,
    "vgg16": 512,
    "vgg19": 512, ## 512, 7, 7

    "vgg11_bn": 512,
    "vgg13_bn": 512,
    "vgg16_bn": 512,
    "vgg19_bn": 512,

    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048, 
    "resnet101": 2048, ##2048, 2, 2
 
    "resnext50_32x4d": 2048,
    "resnext101_32x8d": 2048,

    "wide_resnet50_2": 2048,
    "wide_resnet101_2": 2048,

    "shufflenet_v2_x0_5": 192, ##192, 3, 3
    "shufflenet_v2_x1_0": 464, ##464, 3, 3

    "densenet121": 1024,
    "densenet169": 1664,
    "densenet201": 1920,
    "densenet161": 2208, ##2208, 3, 3

    "mobilenet_v2": 1280, ##1280, 3, 3
    
    "efficientnet_b0": 1280,
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




class dre_extractor_builder(nn.Module):
    def __init__(self, model_name, num_class=23, pretrained=True):
        super(dre_extractor_builder, self).__init__()

        if model_name in feature_dims:
            feature_dim = feature_dims[model_name]
        else:
            raise Exception("Not supported network...")
       
        conv_blocks = load_fn(model_name=model_name, pretrained=pretrained)
        if ("mobilenet" in model_name) or ("densenet" in model_name):
            conv_blocks = list(conv_blocks.children())[:-1]
        else:
            conv_blocks = list(conv_blocks.children())[:-2]
        # conv_blocks += [nn.AvgPool2d((2,2))]
        self.conv = nn.Sequential(*conv_blocks)


        self.extractor1 = nn.Sequential(
            nn.Conv2d(feature_dim, 32**2*3, kernel_size=3, padding=1),
            nn.BatchNorm2d(32**2*3),
            nn.ReLU()
        )
        
        self.extractor2 = nn.Sequential(
            nn.Conv2d(32**2*3, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
                
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        feat = self.conv(x) #7*7*512
        feat = self.extractor1(feat)
        out = self.extractor2(feat)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out, feat.view(feat.size(0), -1)



class vgg8_extract(nn.Module):
    def __init__(self, num_class=23):
        super(vgg8_extract, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'])
        
        self.extractor1 = nn.Sequential(
            nn.Conv2d(512, 32**2*3, kernel_size=3, padding=1),
            nn.BatchNorm2d(32**2*3),
            nn.ReLU()
        )

        self.extractor2 = nn.Sequential(
            nn.Conv2d(32**2*3, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
                
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, num_class),
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
        # layers += [nn.AvgPool2d(kernel_size=3, stride=3)]
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x) #7*7*512
        features = self.extractor1(features)
        out = self.extractor2(features)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out, features.view(features.size(0), -1)


if __name__ == "__main__":
    net = dre_extractor_builder(model_name="efficientnet_b0", num_class=23).cuda()
    net = nn.DataParallel(net)
    x = torch.randn(4,3,224,224).cuda()
    out, feat = net(x)
    print(out.size())
    print(feat.size())
