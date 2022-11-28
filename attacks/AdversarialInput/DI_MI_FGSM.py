from kornia import augmentation as KA
import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
from torchvision import transforms


class DI_MI_FGSM(AdversarialInputAttacker):
    '''
    DI-FGSM is not using data augmentation to increase data for optimizing perturbations.
    DI-FGSM actually is using differentiable data augmentations,
    and this data augmentation can be viewed as a part of model(from SI-FGSM)
    '''

    def __init__(self, model: List[nn.Module], epsilon: float = 16 / 255,
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
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
        self.aug_policy = transforms.Compose([
            transforms.RandomCrop((295, 295), padding=4),
        ])
        super(DI_MI_FGSM, self).__init__()

    def init(self):
        # set the model parameters requires_grad is False
        for model in self.model:
            model.requires_grad_(False)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.requires_grad = True
            aug_x = self.aug_policy(x)
            loss = 0
            for model in self.model:
                loss += self.criterion(model(aug_x), y)
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
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x
