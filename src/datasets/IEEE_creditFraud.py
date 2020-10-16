import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
from .preprocessing import get_target_label_idx
import torch
from torch.utils.data import Subset


# Need to complete preprocessing here



def process_features(features):
    features = np.asarray(features)
    for i, item in enumerate(features):
        if type(item) == 'numpy.str_':
            np.delete(features, i, axis=1)

    print(features)
    return features

def read_nth(reader, row):
    for i, x in enumerate(reader):
        if i == row:
            return list(row)

class IEEE_CreditFraud_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [0,1]
        self.outlier_classes.remove(normal_class)

        transform = None

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyIEEE_CreditFraud(root=self.root, train=True, transform=transform)

        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)

        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyIEEE_CreditFraud(root=self.root, train=False,
                                transform=transform)

class MyIEEE_CreditFraud(Dataset):
    """Torchvision Credit Fraud class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train, transform):

        # Path to the csv file
        path_2_train_identity = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/IEEE_CreditFraud/train_identity.csv"
        path_2_train_transaction = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/IEEE_CreditFraud/train_transaction.csv"

        # Open and read the csv
        # tables = []
        csv_files = [path_2_train_identity, path_2_train_transaction]
        labels = []

        
        with open(csv_files[1], mode='r') as infile:
            reader = csv.reader(infile)
            for i,row in enumerate(reader):
                try:
                    labels.append(float(row[1]))
                except:
                    continue
        
        labels = labels[0:590000]
        self.train_labels = labels[0:500000]
        self.test_labels = labels[500000:590000]

        self.test_labels, self.train_labels = torch.FloatTensor(np.asarray(self.test_labels)), torch.FloatTensor(np.asarray(self.train_labels))
        self.csv_files = csv_files
        # print(tables)
        self.train = train


        self.transform = transform

    def __getitem__(self, index):


        if self.train:
            target, features, reader, transaction_features, identity_features = None, None, None, None, None

            with open(self.csv_files[0], mode='r') as infile:
                reader = csv.reader(infile)
                transaction_features = read_nth(reader, index)

            with open(self.csv_files[1], mode='r') as infile:
                reader = csv.reader(infile)

                identity_features = read_nth(reader, index)
                features, target = transaction_features+identity_features, transaction_features[1]
                del features[1]

        else:
            index = index+500000
            reader, transaction_features, identity_features = None, None

            with open(self.csv_files[2], mode='r') as infile:
                reader = csv.reader(infile)
                transaction_features = read_nth(reader, index)

            with open(self.csv_files[3], mode='r') as infile:
                reader = csv.reader(infile)
                identity_features = read_nth(reader, index)
                features, target = transaction_features+identity_features, transaction_features[1]
                del features[1]
                del features[0]
                features = process_features(features)
                print(features)

        if self.transform:
            return self.transform(features), self.transform(target), index
        else:
            return features, target, index
    
    def __len__(self):

        if self.train:
            return 500000
        return 90000
