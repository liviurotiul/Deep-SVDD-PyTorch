from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .creditFraudNet import CreditFraudNet, CreditFraudNet_Autoencoder
from .malware_detection_net import MalwareDetectionNet, MalwareDetectionNet_Autoencoder
from ._2nHack_network import _2nHackNet

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'credit_fraud_net', 'malware_detection_net', '_2nHack_net')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'credit_fraud_net':
        net = CreditFraudNet()

    if net_name == 'malware_detection_net':
        net = MalwareDetectionNet()
    
    if net_name == '_2nHack_net':
        net = _2nHackNet()
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'credit_fraud_net', 'malware_detection_net')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'credit_fraud_net':
        ae_net = CreditFraudNet_Autoencoder()
    
    if net_name == 'malware_detection_net':
        ae_net = MalwareDetectionNet_Autoencoder()

    return ae_net
