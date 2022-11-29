'''
PGD: Projected Gradient Descent
'''

import torch
from torch import nn
from typing import Callable, List
from attacks.utils import *
from .AdversarialInputBase import AdversarialInputAttacker


class PGD(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module], epsilon: float = 16 / 255,
                 total_step: int = 10, random_start: bool = True,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss().to(
                     torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                 targeted_attack=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ):
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.device = device
        super(PGD, self).__init__(model)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):

        original_x = x.clone()
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.requires_grad = True
            loss = 0
            for model in self.models:
                loss += self.criterion(model(x.to(model.device)), y.to(model.device)).to(x.device)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            # update
            if self.targerted_attack:
                x -= self.step_size * grad.sign()
            else:
                x += self.step_size * grad.sign()
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x
