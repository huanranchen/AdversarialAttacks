import torch
from .AdversarialInputBase import AdversarialInputAttacker
from typing import Callable, List, Iterable
from attacks.utils import *
from .utils import cosine_similarity
from torch import nn


class MI_CosineSimilarityEncourager(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 epsilon: float = 16 / 255,
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 outer_optimizer=None,
                 ):
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        super(MI_CosineSimilarityEncourager, self).__init__(model)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                grad = x.grad
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.targerted_attack:
                    momentum = self.mu * momentum - grad / torch.norm(grad, p=1)
                    x += self.step_size * momentum.sign()
                else:
                    momentum = self.mu * momentum + grad / torch.norm(grad, p=1)
                    x += self.step_size * momentum.sign()
                    # x += self.step_size * grad.sign()
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 10):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = now
        if self.outer_optimizer is None:
            fake_grad = (patch - self.original)
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            patch.mul_(0)
            patch.add_(self.original)
            patch.add_(ksi * self.outer_momentum.sign())
            # patch.add_(ksi * fake_grad)
        else:
            fake_grad = - ksi * (patch - self.original)
            self.outer_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original)
            patch.grad = fake_grad
            self.outer_optimizer.step()
        patch = clamp(patch)

        grad_similarity = cosine_similarity(self.grad_record)
        del self.grad_record
        del self.original
        return patch
