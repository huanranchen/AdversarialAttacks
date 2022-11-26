from typing import Callable, List, Iterable
from torch import nn
from attacks.utils import *
from .PerturbationObject import Perturbation


class ParallelAttacker():
    '''
    please set your learning rate in optimizer
    set data augmentation in your loader.
    '''

    def __init__(self,
                 models: List[nn.Module],
                 perturbation: Perturbation,
                 transformation: nn.Module = nn.Identity(),
                 criterion: Callable = nn.CrossEntropyLoss(),
                 **kwargs,
                 ):
        self.perturbation = torch.randn
        self.models = models
        perturbation.requires_grad(True)
        self.perturbation = perturbation
        self.transform = transformation
        self.criterion = criterion

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

                loss = 0
                for model in self.models:
                    loss += self.criterion(model(x.to(model.device)), y.to(model.device))
                self.perturbation.zero_grad()
                loss.backward()
                self.perturbation.step()

                iter_step += 1
                if iter_step > total_iter_step:
                    self.perturbation.requires_grad(False)
                    return self.perturbation
