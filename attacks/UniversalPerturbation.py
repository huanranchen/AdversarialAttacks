from .base import BaseAttacker
from typing import Callable, List
from optimizer import Optimizer
from torch import nn
from .utils import *


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


class UniversalPerturbation(BaseAttacker):
    '''
    please set your learning rate in optimizer
    set data augmentation in your loader.
    '''

    def __init__(self,
                 models: List[nn.Module],
                 perturbation: Perturbation,
                 optimizer: Callable,
                 transformation: nn.Module = nn.Identity(),
                 criterion: Callable = nn.CrossEntropyLoss(),
                 ):
        self.perturbation = torch.randn
        self.models = models
        perturbation.requires_grad(True)
        self.perturbation = perturbation
        self.optimizer = optimizer(perturbation.perturbation)
        self.transform = transformation
        self.criterion = criterion
        super(UniversalPerturbation, self).__init__()

    def init(self):
        for model in self.models:
            model.requires_grad_(False)

    def attack(self, loader: DataLoader,
               total_iter_step: int = 1000,
               is_clamp=True):
        iter_step = 0
        while True:
            for x, y in loader:
                x = x + self.perturbation.perturbation
                if is_clamp:
                    x = clamp(x)
                x = self.transform(x)
                loss: torch.tensor = 0
                for model in self.models:
                    loss += self.criterion(model(x), y)
                loss.backward()
                self.optimizer.step()

                iter_step += 1
                if iter_step > total_iter_step:
                    self.perturbation.requires_grad(False)
                    return self.perturbation
