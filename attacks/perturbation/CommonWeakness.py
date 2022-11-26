from .ParallelAttacker import ParallelAttacker
from typing import Callable, List, Iterable
from torch import nn
from attacks.utils import *
from .utils import cosine_similarity


class CosineSimilarityEncourager(ParallelAttacker):
    def __init__(self, *args, **kwargs):
        super(CosineSimilarityEncourager, self).__init__(*args, **kwargs)
        if kwargs['outer_optimizer'] is not None:
            self.outer_optimizer = kwargs['outer_optimizer'](self.perturbation.perturbation)
        else:
            self.outer_optimizer = None

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
                    loss += self.criterion(model(x), y)
                    self.perturbation.zero_grad()
                    loss.backward()
                    self.perturbation.step()

                iter_step += 1
                if iter_step > total_iter_step:
                    self.perturbation.requires_grad(False)
                    return self.perturbation

    @torch.no_grad()
    def begin_attack(self):
        self.original = self.perturbation.perturbation.clone()
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, ksi=1):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = self.perturbation.perturbation
        if self.outer_optimizer is None:
            patch.mul_(ksi)
            patch.add_((1 - ksi) * self.original)
            self.original_patch = None
        else:
            fake_grad = - ksi * (patch - self.original_patch)
            self.outer_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original_patch)
            patch.grad = fake_grad
            self.outer_optimizer.step()

        grad_similarity = cosine_similarity(self.grad_record)
        del self.grad_record
        del self.original
