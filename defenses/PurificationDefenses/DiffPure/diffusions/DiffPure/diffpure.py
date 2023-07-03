import torch
from torch import nn
from models import WideResNet_70_16_dropout
from defenses.PurificationDefenses.DiffPure.diffusions.sampler.ddim import DDIM
from defenses.PurificationDefenses.DiffPure.diffusions.sampler.sde import DiffusionSde, DiffusionOde


class DiffusionPure(nn.Module):
    def __init__(self, mode='sde',
                 pre_transforms=nn.Identity(),
                 post_transforms=nn.Identity(),
                 model=None,
                 *args, **kwargs):
        super(DiffusionPure, self).__init__()
        self.device = torch.device('cuda')
        if mode == 'sde':
            self.diffusion = DiffusionSde(*args, **kwargs)
        elif mode == 'ode':
            self.diffusion = DiffusionOde(*args, **kwargs)
        elif mode == 'ddim':
            self.diffusion = DDIM(*args, **kwargs)
        if model is None:
            model = WideResNet_70_16_dropout()
        self.model = model
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.init()

    def init(self):
        self.eval().requires_grad_(False)
        self.to(self.device)

    def forward(self, x, *args, **kwargs):
        x = self.pre_transforms(x)
        x = self.diffusion(x, *args, **kwargs)
        x = self.post_transforms(x)
        x = self.model(x)
        return x


class SerialDiffusionPure(DiffusionPure):
    def __init__(self, *args, **kwargs):
        self.t = kwargs['diffusion_number']
        del kwargs['diffusion_number']
        kwargs['dt'] = 1e-3
        super(SerialDiffusionPure, self).__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        x = self.pre_transforms(x)
        for _ in range(self.t):
            x = self.diffusion(x, *args, **kwargs)
        x = self.post_transforms(x)
        x = self.model(x)
        return x
