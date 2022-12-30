import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
from torchvision import transforms
from .utils import *


class VMI_FGSM(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 epsilon: float = 16 / 255,
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 ):
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(VMI_FGSM, self).__init__(model)

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
            # x.requires_grad = True
            # logit = 0
            # for model in self.models:
            #     logit += model(x.to(model.device)).to(x.device)
            # loss = self.criterion(logit, y)
            # loss.backward()
            # grad = x.grad
            # x.requires_grad = False
            # update
            grad = self.calculate_v(x, y)
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    def calculate_v(self, x: torch.tensor, y: torch.tensor, N=20, beta=1.5):
        v = torch.zeros_like(x)
        ranges = beta * self.epsilon
        for _ in range(N):
            now = x + (torch.rand_like(x) - 0.5) * 2 * ranges
            now.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(now.to(model.device)).to(now.device)
            loss = self.criterion(logit, y)
            loss.backward()
            v += now.grad
        v /= N
        return v


class VMI_Inner_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 epsilon: float = 16 / 255,
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 outer_optimizer=None,
                 reverse_step_size=16 / 255 / 15,
                 inner_step_size: float = 250,
                 DI=False,
                 ):
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        self.reverse_step_size = reverse_step_size
        super(VMI_Inner_CommonWeakness, self).__init__(model)
        self.inner_step_size = inner_step_size
        self.DI = DI
        if DI:
            self.aug_policy = transforms.Compose([
                transforms.RandomCrop((int(224 * 0.9), int(224 * 0.9)), padding=224 - int(224 * 0.9)),
            ])
        else:
            self.aug_policy = nn.Identity()

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            self.begin_attack(x.clone().detach())
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()
                # x -= self.reverse_step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
            # --------------------------------------------------------------------------------#
            # second step
            x.grad = None
            # self.begin_attack(x.clone().detach())
            for model in self.models:
                grad = self.calculate_v(x, y, model)
                self.grad_record.append(grad)
                # update
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
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
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
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

    def calculate_v(self, x: torch.tensor, y: torch.tensor, model: nn.Module, N=20, beta=1.5):
        v = torch.zeros_like(x)
        ranges = beta * self.epsilon
        for _ in range(N):
            now = x + (torch.rand_like(x) - 0.5) * 2 * ranges
            now.requires_grad = True
            logit = model(now.to(model.device)).to(now.device)
            loss = self.criterion(logit, y)
            loss.backward()
            v += now.grad
        v /= N
        return v
