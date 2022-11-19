import torch
from torch.optim import Optimizer


class FGSM(Optimizer):
    def __init__(self, params, lr, ):
        dampening = 0
        weight_decay = 0
        nesterov = False
        maximize = False
        momentum = 0
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super(FGSM, self).__init__(params, defaults)
        self.lr = lr

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.add_(-self.lr * p.grad.sign())
