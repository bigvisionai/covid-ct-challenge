import torch
import torch.nn as nn
from torchvision import models
import os

class CovidCT(nn.Module):
    """Covid CT detector, uses pretrained Resnet as a backbone to extract features
    """
    
    def __init__(self): # add conf file

        super(CovidCT,self).__init__()

        # init backbone
        self.backbone = self._generate_resnet()

        # Need to modify DropOuts
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=1024,out_features=2),
        )

    def forward(self,x):
        x = self.backbone(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def _generate_resnet(self):

        backbone = models.resnet50(pretrained=True)
        resnet_modules = list(backbone.children())
        body = nn.Sequential(*resnet_modules[:-1])

        for x in body.parameters():
            x.requires_grad = False

        # for x in body[5].parameters():
        #     x.requires_grad = True

        # for x in body[6].parameters():
        #     x.requires_grad = True

        for x in body[7][2].parameters():
            x.requires_grad = True
        
        for x in body[8].parameters():
            x.requires_grad = True
        
        return body
 