import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ReIDModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet50", feat_dim=512):
        super().__init__()

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
        feat = self.gap(feat).view(feat.size(0), -1)
        feat = self.fc(feat)
        feat_bn = self.bnneck(feat)

        logits = self.classifier(feat_bn)

        if return_feature:
            return F.normalize(feat_bn, p=2, dim=1)

        return logits, feat_bn
