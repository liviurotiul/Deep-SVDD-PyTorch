from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .creditFraud import CreditFraud_Dataset
from .IEEE_creditFraud import IEEE_CreditFraud_Dataset
from .malware_detection import MalwareDetection_Dataset
from ._2nHack_dataset import _2nHack_Dataset

def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'creditFraud', 'IEEE_creditFraud', "malware_detection", '_2nHack')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'creditFraud':
        dataset = CreditFraud_Dataset(root=data_path)

    if dataset_name == 'IEEE_creditFraud':
        dataset = IEEE_CreditFraud_Dataset(root=data_path)
    
    if dataset_name == 'malware_detection':
        dataset = MalwareDetection_Dataset(root=data_path)

    if dataset_name == '_2nHack':
        dataset = _2nHack_Dataset(root=data_path)

    return dataset
