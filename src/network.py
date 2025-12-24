import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


######### ATTENTION MECHANISM (CBAM) #########
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

######### ATTENTION MECHANISM #########

class ReIDModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet50", feat_dim=512, use_attention=False):
        super().__init__()

        #use attention
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = CBAM(in_dim)

        #choose backbone (CNN for img classification)
        if backbone=="resnet18":
            base = models.resnet18(pretrained=True)
            in_dim = 512
        elif backbone=="resnet50":
            base = models.resnet50(pretrained=True)
            in_dim = 2048
        else:
            raise ValueError("backbone must be resnet18 or resnet50")

        self.backbone = nn.Sequential(*list(base.children())[:-2])
        

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_dim, feat_dim)
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x, return_feature=False):
        feat = self.backbone(x)
        if self.use_attention:
            feat = self.attention(feat)
            
        feat = self.gap(feat).view(feat.size(0), -1)
        feat = self.fc(feat)
        feat_bn = self.bnneck(feat)

        logits = self.classifier(feat_bn)

        if return_feature:
            return F.normalize(feat_bn, p=2, dim=1)

        return logits, feat_bn
