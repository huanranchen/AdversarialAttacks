import torch
from attacks.utils import *
from torch import nn
from typing import Callable
from .base import BaseAttacker


class MI_FGSM(BaseAttacker):
    def __init__(self, model: nn.Module, epsilon: float = 16 / 255,
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 5e-3,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 ):
        self.model = model
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        
        super(MI_FGSM, self).__init__()

    def init(self):
        # set the model parameters requires_grad is False
        self.model.requires_grad_(False)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        momentum = torch.zeros_like(x)
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
                momentum = self.mu * momentum - grad / torch.norm(grad, p=1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad, p=1)
                x += self.step_size * momentum.sign()
            x = clamp(x)

        return x
