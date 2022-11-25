'''
PGD: Projected Gradient Descent
'''

import torch
from torch import nn
from typing import Callable
from attacks.utils import *
from .base import BaseAttacker


class PGD(BaseAttacker):
    def __init__(self, model: nn.Module, epsilon: float = 16 / 255,
                 total_step: int = 10, random_start: bool = True,
                 step_size: float = 5e-3,
                 criterion: Callable = nn.CrossEntropyLoss().to(
                     torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                 targeted_attack=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ):
        self.model = model
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.device = device
        super(PGD, self).__init__()

    def init(self):
        # set the model parameters requires_grad is False
        self.model.requires_grad_(False)

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
            loss = self.criterion(self.model(x), y)
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