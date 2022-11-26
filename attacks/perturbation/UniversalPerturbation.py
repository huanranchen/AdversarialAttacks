from attacks.AdversarialInput.AdversarialInputBase import AdversarialInputAttacker
from typing import Callable, List, Iterable
from torch import nn
from attacks.utils import *


class Perturbation():
    def __init__(self,
                 perturbation_size: tuple = (3, 224, 224),
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.perturbation = torch.zeros(perturbation_size, device=device)
        self.device = device

    def gaussian_init(self, is_clamp=True, scale=0.5, mean=0.5):
        self.perturbation = torch.randn_like(self.perturbation, device=self.device) * scale + mean
        if is_clamp:
            self.perturbation = clamp(self.perturbation)

    def uniform_init(self):
        self.perturbation = torch.rand_like(self.perturbation, device=self.device)

    def constant_init(self, constant=0):
        self.perturbation = torch.zeros_like(self.perturbation, device=self.device) + constant

    def requires_grad(self, requires_grad: bool = True):
        self.perturbation.requires_grad = requires_grad


class UniversalPerturbation():
    '''
    please set your learning rate in optimizer
    set data augmentation in your loader.
    '''

    def __init__(self,
                 attacker: AdversarialInputAttacker,
                 models: List[nn.Module],
                 perturbation: Perturbation,
                 transformation: nn.Module = nn.Identity(),
                 criterion: Callable = nn.CrossEntropyLoss(),
                 ):
        self.perturbation = torch.randn
        self.models = models
        perturbation.requires_grad(True)
        self.perturbation = perturbation
        self.transform = transformation
        self.criterion = criterion
        self.attacker = attacker
        super(UniversalPerturbation, self).__init__()

    def init(self):
        for model in self.models:
            model.requires_grad_(False)

    def tensor_to_loader(self, x, y):
        return [(x, y)]

    def attack(self,
               loader: DataLoader or Iterable,
               total_iter_step: int = 1000,
               is_clamp=True):
        iter_step = 0
        while True:
            for x, y in loader:
                x = x + self.perturbation.perturbation
                if is_clamp:
                    x = clamp(x)
                x = self.transform(x)

                self.attacker.attack(x, y)

                iter_step += 1
                if iter_step > total_iter_step:
                    self.perturbation.requires_grad(False)
                    return self.perturbation
