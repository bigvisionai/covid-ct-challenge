import torch
import torch.nn as nn
from torchvision import models
import os

class CovidCT(nn.Module):
    """Covid CT detector, uses pretrained alexnet as a backbone to extract features
    """
    
    def __init__(self): # add conf file

        super(CovidCT,self).__init__()

        # init backbone
        self.backbone = models.alexnet(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Need to modify DropOuts
        self.fc = nn.Sequential(
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=64,out_features=2)
        )

    def forward(self,x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0),256)
        x = self.fc(x)
        return x
 