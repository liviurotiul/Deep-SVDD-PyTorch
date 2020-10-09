import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
from .preprocessing import get_target_label_idx
import torch
from torch.utils.data import Subset

class CreditFraud_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [0,1]
        self.outlier_classes.remove(normal_class)

        transform = None

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCreditFraud(root=self.root, train=True, transform=transform)

        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)

        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCreditFraud(root=self.root, train=False,
                                transform=transform)

class MyCreditFraud(Dataset):
    """Torchvision Credit Fraud class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train, transform):


        # Path to the csv file
        path_2_csv = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/CreditFraud/creditcard.csv"

        # Open and read the csv
        with open(path_2_csv, mode='r') as infile:
            reader = csv.reader(infile)
            table = []
            for row in reader:
                table.append(list(row))

        table = table[1:-1]        
        table = [[float(x) for x in row] for row in table]

        table = np.asarray(table)
        features = table[:,0:30]
        labels = table[:,-1]
        features[:,0] = features[:,0]/np.max(features[:,0])
        # print(features[])

        # self.train_data = torch.from_numpy(features[0:200000])
        # self.test_data = torch.from_numpy(features[200001:284807])
        # self.train_labels = torch.from_numpy(labels[0:200000])
        # self.test_labels = torch.from_numpy(labels[200001:284807])

        self.train_data = torch.from_numpy(features[0:200000])
        self.test_data = torch.from_numpy(features[0:200000])
        self.train_labels = torch.from_numpy(labels[0:200000])
        self.test_labels = torch.from_numpy(labels[0:200000])

        self.train = train
        self.transform = transform

    def __getitem__(self, index):

        if self.train:
            features, target = self.train_data[index], self.train_labels[index]
        else:
            features, target = self.test_data[index], self.test_labels[index]


        if self.transform:
            return self.transform(features), self.transform(target), index
        else:
            return features, target, index
    
    def __len__(self):

        if self.train:
            return len(self.train_data)
        return len(self.test_data)
