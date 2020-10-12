import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
from .preprocessing import get_target_label_idx
import torch
from torch.utils.data import Subset


# Need to complete preprocessing here

def process_identity_table(table):
    # table = table[0:39]
    # table = np.asarray(table)
    print(table)

def process_transaction_table(table):
    # table = table[0:39]
    # table = np.asarray(table)
    print(table)

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
        path_2_test_identity = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/IEEE_CreditFraud/test_identity.csv"
        path_2_test_transaction = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/IEEE_CreditFraud/test_transaction.csv"
        path_2_train_identity = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/IEEE_CreditFraud/train_identity.csv"
        path_2_train_transaction = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/IEEE_CreditFraud/train_transaction.csv"

        # Open and read the csv
        # tables = []
        csv_files = [path_2_test_identity, path_2_test_transaction, path_2_train_identity, path_2_train_transaction]


        
        with open(csv_files[1], mode='r') as infile:
            reader = csv.reader(infile)
            self.test_labels = []
            for i,row in enumerate(reader):
                # print(row)
                self.test_labels.append(row[1])

        with open(csv_files[3], mode='r') as infile:
            reader = csv.reader(infile)
            self.train_labels = []
            for i,row in enumerate(reader):
                # print(row)
                self.train_labels.append(row[1])

        self.test_labels, self.train_labels = torch.FloatTensor(np.asarray(self.test_labels)), torch.FloatTensor(np.asarray(self.train_labels))
        self.csv_files = csv_files
        # print(tables)



        self.transform = transform

    def __getitem__(self, index):

        #TODO: preprocesing
        if self.train:
            reader, transaction_features, identity_features = None, None

            with open(self.csv_file[2], mode='r') as infile:
                reader = csv.reader(infile)
            
            transaction_features = read_nth(reader, index)

            with open(self.csv_file[3], mode='r') as infile:
                reader = csv.reader(infile)

            identity_features = read_nth(reader, index)
            features, target = transaction_features+identity_features, transaction_features[1]
            del features[1]
        else:
            reader, transaction_features, identity_features = None, None

            with open(self.csv_file[0], mode='r') as infile:
                reader = csv.reader(infile)
            transaction_features = read_nth(reader, index)

            with open(self.csv_file[1], mode='r') as infile:
                reader = csv.reader(infile)

            identity_features = read_nth(reader, index)
            features, target = transaction_features+identity_features, transaction_features[1]
            del features[1]


        if self.transform:
            return self.transform(features), self.transform(target), index
        else:
            return features, target, index
    
    def __len__(self):

        if self.train:
            return len(self.train_data)
        return len(self.test_data)
