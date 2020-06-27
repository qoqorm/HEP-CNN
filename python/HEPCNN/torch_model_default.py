#!/usr/bin/env python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, width, height, **kwargs):
        super(MyModel, self).__init__()
        model = "default" if "model" not in kwargs else kwargs["model"]
        self.fw = width//2//2//2
        self.fh = height//2//2//2

        self.nch = 5 if '5ch' in model else 3
        self.doLog = ('log' in model)
        if 'norm0' in model: self.doNorm = 0 ## do not normalize at all
        elif 'norm1' in model: self.doNorm = 111 ## normalize all, 111
        else: self.doNorm = 101 ## The default normalization: ecal and tracker
        self.doCat = ('cat' in model)

        self.conv = nn.Sequential(
            nn.Conv2d(self.nch, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, kernel_size=(2, 2), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, kernel_size=(2, 2), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            #nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fw*self.fh*256 + (3 if self.doCat else 0), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        n, c = x.shape[0], x.shape[1]
        if c > 6: ## We don't expect image more than 6 channel, this indicates that the image format was NHWC.
            x = x.permute(0,3,1,2)
            c = x.shape[1]
        s, _ = torch.max(x.view(n,c,-1), dim=-1)
        if self.doNorm %   10 != 0: x[:,0,:,:] /= s[:,0,None,None]
        if self.doNorm %  100 != 0: x[:,1,:,:] /= s[:,1,None,None]
        if self.doNorm % 1000 != 0: x[:,2,:,:] /= s[:,2,None,None]
        if self.nch == 5:
            xx = x[:,:2,:,:]
            x = torch.cat((x, xx), dim=1)
        if self.doLog:
            x[:,:2,:,:] = torch.log10(x[:,:2,:,:]/1e-5+1)
        x = self.conv(x)
        x = x.view(-1, self.fw*self.fh*256)
        if self.doCat: x = torch.cat([x, s], dim=-1)
        x = self.fc(x)
        return x

