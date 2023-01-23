from data import get_NIPS17_loader
from tester import test_autoattack_acc
from defenses import DiffusionPureImageNet
test_autoattack_acc(DiffusionPureImageNet(), loader=get_NIPS17_loader())



