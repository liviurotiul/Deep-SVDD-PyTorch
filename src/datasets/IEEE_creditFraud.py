import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
from .preprocessing import get_target_label_idx
import torch
from torch.utils.data import Subset


# Need to complete preprocessing here
def process_tables(tables):
    identity_tables = [tables[0], tables[2]]
    transcation_tables =  [tables[1], tables[3]]
    identity_tables = [process_identity_table(x) for x in identity_tables]
    transcation_tables = []
    return 

def process_identity_table(table):
    table = table[0:39]
    table = np.asarray(table)
    print(table)

def process_transaction_table(table):
    table = table[0:39]
    table = np.asarray(table)
    print(table)

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
        tables = []
        csv_files = [path_2_test_identity, path_2_test_transaction, path_2_train_identity, path_2_train_transaction]
        for csv_file in csv_files:
            with open(csv_file, mode='r') as infile:
                reader = csv.reader(infile)
                table = []
                for i,row in enumerate(reader):
                    # print(row)
                    table.append(list(row))
                    if i == 100000:
                        break
            tables.append(table)

        self.csv_files = csv_files
        # print(tables)

        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        
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
