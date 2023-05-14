import torch
import torch.nn as nn


feature_dims = {
    
    "vgg8": 512,
    "vgg11": 512,
    "vgg13": 512,
    "vgg16": 512, 
    "vgg19": 512, 
    "resnet18": 512, 
    "resnet34": 512, 
    "resnet50": 2048, 
    "resnet101": 2048, 
    "densenet121": 1024, ##1024, 3, 3
    "densenet169": 1664, 
    "densenet201": 1920, 
    "densenet161": 2208, ##2208, 3, 3
    "mobilenet_v2": 1280,
    
    "resnet8": 64,
    "resnet14": 64,
    "wrn_16_1": 64,
    "wrn_40_1": 64,
    "shufflenet_v2_x0_5": 192, 
    "shufflenet_v2_x1_0": 464, 
    
}


class hint_regressor(nn.Module):
    def __init__(self, t_name, s_name):
        super(hint_regressor, self).__init__()
        
        self.input_C_dim = feature_dims[s_name]
        self.output_C_dim = feature_dims[t_name]
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.input_C_dim, self.output_C_dim, 3, 1, padding=1),
            nn.BatchNorm2d(self.output_C_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        return out



if __name__ == '__main__':
    import torch

    x = torch.randn(10, feature_dims["shufflenet_v2_x0_5"], 3, 3)
    net = hint_regressor("vgg16", "shufflenet_v2_x0_5")
    out = net(x)
    print(out.shape)