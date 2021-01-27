import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class _2nHackNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 10
        self.max_pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, bias=False)
        self.fc1 = nn.Linear(1152, 100, bias=False)
        self.fc2 = nn.Linear(100, 10, bias=False)


    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)
        x = self.bn1(self.max_pool(F.leaky_relu(self.conv1(x))))
        x = self.bn2(self.max_pool(F.leaky_relu(self.conv2(x))))
        x = self.bn3(self.max_pool(F.leaky_relu(self.conv3(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class _2nHackNet_Autoencoder(BaseNet):
#net3
    def __init__(self):
        super().__init__()
        self.hidden_dim = 24
        self.rep_dim = 6

        #encoder
        self.fc1 = nn.Linear(55, self.hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(self.hidden_dim, 6, bias=False)


        #decoder
        self.fc3_b = nn.Linear(self.hidden_dim, 55, bias=False)
        self.bn3_b = nn.BatchNorm1d(self.hidden_dim, eps=1e-04, affine=False)
        self.fc4_b = nn.Linear(6, self.hidden_dim, bias=False)

    def forward(self, x):
        x = x.float()
        x.unsqueeze(-1)

        x = self.bn1(F.leaky_relu(self.fc1(x)))
        x = self.fc2(x)

        x = self.bn3_b(F.leaky_relu(self.fc4_b(x)))
        x = self.fc3_b(x)


        return x