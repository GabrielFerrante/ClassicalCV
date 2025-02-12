import torch
import torch.nn as nn
from torchvision.models import vgg19

class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg19(pretrained=True).features[:35].eval()
        self.mse = nn.MSELoss()

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)       # Formato (B, C, H*W)
        gram = torch.bmm(features, features.transpose(1, 2))  # Multiplicação (B, C, C)
        return gram / (c * h * w)   

    def forward(self, input, target):
        input_gram = self.gram_matrix(self.model(input))
        target_gram = self.gram_matrix(self.model(target))
        return self.mse(input_gram, target_gram)

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg19(pretrained=True).features[:35].eval()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(self.model(input), self.model(target))

class TotalVariationLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        tv_h = torch.pow(x[:,:,1:,:]-x[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,:,1:]-x[:,:,:,:-1], 2).sum()
        return (tv_h + tv_w) / (batch_size * h_x * w_x)