import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker


class MI_SAM(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module], epsilon: float = 16 / 255,
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 10,
                 ):
        self.models = model
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM, self).__init__(model)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            x.requires_grad = True
            loss = 0
            for model in self.models:
                loss += self.criterion(model(x.to(model.device)), y.to(model.device)).to(x.device)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()

            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            loss = 0
            for model in self.models:
                loss += self.criterion(model(x.to(model.device)), y.to(model.device)).to(x.device)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x
