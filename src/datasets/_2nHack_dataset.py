import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
from base.torchvision_dataset import TorchvisionDataset
import torchvision.transforms as transforms
from .preprocessing import get_target_label_idx
import torch
from torch.utils.data import Subset
import pickle
import torchvision

class _2nHack_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [0,1]
        self.outlier_classes.remove(normal_class)

        transform = None

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = My_2nHack(root=self.root, train=True, transform=transform)

        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(torch.from_numpy(np.array(train_set.train_labels)).clone().data.cpu().numpy(), self.normal_classes)

        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = My_2nHack(root=self.root, train=False,
                                transform=transform)


class My_2nHack(Dataset):
    """Torchvision Credit Fraud class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, train, transform):


        # Path to the csv file
        image_test_path = "/home/liviu/Documents/Dev/2nHack/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_test_dict.pkl"
        image_train_path = "/home/liviu/Documents/Dev/2nHack/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_train_dict.pkl"
        label_path = "/home/liviu/Documents/Dev/2nHack/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/label_dict.pkl"

        image_dict = {}

        with open(image_train_path, "rb") as image_train_dict:
            image_dict = pickle.load(image_train_dict)

        with open(image_test_path, "rb") as image_test_dict:
            image_dict.update(pickle.load(image_test_dict))


        label_dict = open(label_path, "rb")
        label_dict = pickle.load(label_dict)

        table = []
        features = []
        labels = []

        for key in image_dict:
            if key == 'X_ray_image_name':
                continue
            try:
                temp = label_dict[key]
            except:
                continue

            image = image_dict[key]

            label = None
            if temp[2] == 'COVID-19':
                label = 1
                # print(temp)
            else:
                label = 0

            table.append([label, image])
        image_dict = None
        label_dict = None
        if train:
            np.random.shuffle(table)
        features = [row[1] for row in table]

        labels = [row[0] for row in table]
        table = None

        for i in range(0,1700):
            if labels[i] == 1:
                labels[i], labels[i+4000] = labels[i+4000], labels[i]
                features[i], features[i+4000] = features[i+4000], features[i]
                print("swaped")
        # features = [x.convert('LA') for x in features]
        if train:
            self.train_data = features[0:4000]
            self.train_labels = labels[0:4000]
        else:
            self.test_data = features[4000:5911]
            self.test_labels = labels[4000:5911]
        # import pdb; pdb.set_trace()
        # if train:
        #     self.train_data = features[0:5]
        #     self.train_labels = labels[0:5]
        # else:
        #     self.test_data = features[5:9]
        #     self.test_labels = labels[5:9]
        features, labels = None, None
        self.train = train
        self.transform = transform

    def __getitem__(self, index):

        if self.train:
            features, target = torch.from_numpy(np.array(self.train_data[index].convert('L'))), torch.from_numpy(np.array(self.train_labels[index]))

        else:
            features, target = torch.from_numpy(np.array(self.test_data[index].convert('L'))), torch.from_numpy(np.array(self.test_labels[index]))

        # torch.from_numpy(np.array(torchvision.transforms.functional.to_grayscale(features, num_output_channels=1)))
        # print(features.size())
        if self.transform:
            return self.transform(features), self.transform(target), index
        else:
            return features, target, index
    
    def __len__(self):

        if self.train:
            return len(self.train_data)
        return len(self.test_data)
