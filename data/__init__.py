from .cifar import get_CIFAR100_test, get_CIFAR100_train, get_CIFAR10_train, get_CIFAR10_test
from .someset import SomeDataSet, get_someset_loader

__all__ = ['get_CIFAR100_test', 'get_CIFAR100_train', 'get_CIFAR10_test', 'get_CIFAR10_train',
           'SomeDataSet', 'get_someset_loader'
           ]
