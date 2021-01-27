import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


# class CreditFraudNet(BaseNet):
# #net3
#     def __init__(self):
#         super().__init__()
#         self.rep_dim = 6

#         self.fc1 = nn.Linear(30, 21, bias=False)
#         self.bn1 = nn.BatchNorm1d(21, eps=1e-04, affine=False)
#         self.fc2 = nn.Linear(21, 21, bias=False)
#         self.bn2 = nn.BatchNorm1d(21, eps=1e-04, affine=False)

#         self.fc4 = nn.Linear(21, 6, bias=False)

#     def forward(self, x):
#         x = x.float()
#         x.unsqueeze(-1)

#         x = self.bn1(F.leaky_relu(self.fc1(x)))
#         x = self.bn2(F.leaky_relu(self.fc2(x)))

#         x = self.fc4(x)

#         return x

# class CreditFraudNet_Autoencoder(BaseNet):
# #net3
#     def __init__(self):
#         super().__init__()
#         self.rep_dim = 6

#         #encoder
#         self.fc1 = nn.Linear(30, 21, bias=False)
#         self.bn1 = nn.BatchNorm1d(21, eps=1e-04, affine=False)
#         self.fc2 = nn.Linear(21, 21, bias=False)
#         self.bn2 = nn.BatchNorm1d(21, eps=1e-04, affine=False)

#         self.fc4 = nn.Linear(21, 6, bias=False)


#         #decoder
#         self.fc1_b = nn.Linear(21, 30, bias=False)
#         self.bn1_b = nn.BatchNorm1d(21, eps=1e-04, affine=False)
#         self.fc2_b = nn.Linear(21, 21, bias=False)
#         self.bn2_b = nn.BatchNorm1d(21, eps=1e-04, affine=False)

#         self.fc4_b = nn.Linear(6, 21, bias=False)

#     def forward(self, x):
#         x = x.float()
#         x.unsqueeze(-1)

#         x = self.bn1(F.leaky_relu(self.fc1(x)))
#         x = self.bn2(F.leaky_relu(self.fc2(x)))

#         x = self.fc4(x)

#         x = self.bn2_b(F.leaky_relu(self.fc4_b(x)))
#         x = self.bn1_b(F.leaky_relu(self.fc2_b(x)))
#         x = self.fc1_b(x)

#         return x













# python main.py creditFraud credit_fraud_net ../log/mnist_test ../data --objective one-class --lr 0.00003 --n_epochs 50 --lr_milestone 10 --batch_size 500 --weight_decay 0.5e-5 --pretrain True --ae_lr 0.001 --ae_n_epochs 40 --ae_lr_milestone 5






class CreditFraudNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.hidden_dim = 24
        self.rep_dim = 6

        self.fc1 = nn.Linear(30, self.hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(self.hidden_dim, 6, bias=False)


    def forward(self, x):

        x = x.float()
        x.unsqueeze(-1)
        x = self.bn1(F.leaky_relu(self.fc1(x)))
        x = self.fc2(x)

        return x

class CreditFraudNet_Autoencoder(BaseNet):
#net3
    def __init__(self):
        super().__init__()
        self.hidden_dim = 24
        self.rep_dim = 6

        #encoder
        self.fc1 = nn.Linear(30, self.hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(self.hidden_dim, 6, bias=False)


        #decoder
        self.fc3_b = nn.Linear(self.hidden_dim, 30, bias=False)
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
