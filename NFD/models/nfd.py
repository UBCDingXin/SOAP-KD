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

class adapter(nn.Module):
    def __init__(self, t_name, s_name):
        super(adapter, self).__init__()
        
        self.input_C_dim = feature_dims[s_name]
        self.output_C_dim = feature_dims[t_name]
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.input_C_dim, self.output_C_dim, 3, 1, padding=1),
            nn.BatchNorm2d(self.output_C_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        if self.input_C_dim!=self.output_C_dim:
            features = self.conv_blocks(x)
        else:
            features = x
        return features




class reconstruct(nn.Module):
    def __init__(self, t_name, mid_dim=1280):
        super(reconstruct, self).__init__()
        
        self.input_dim = feature_dims[t_name]
        self.mid_dim = mid_dim
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.input_dim, mid_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            
            nn.Conv2d(mid_dim, self.input_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        return features
        




if __name__ == '__main__':
    import torch

    x = torch.randn(10, feature_dims["shufflenet_v2_x0_5"], 3, 3)
    net_adapter = adapter("mobilenet_v2", "shufflenet_v2_x0_5")
    out = net_adapter(x)
    print(out.shape)
    
    net_recons = reconstruct("mobilenet_v2", 1280)
    out = net_recons(out)
    print(out.shape)
    