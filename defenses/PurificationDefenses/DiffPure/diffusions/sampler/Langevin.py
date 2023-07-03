import torch
from torch import nn
from torchvision import transforms
from torch import Tensor
from defenses.PurificationDefenses.DiffPure.diffusions.model import get_unet
from copy import deepcopy


class Langevin(nn.Module):
    def __init__(self, unet: nn.Module = None,
                 mode='cifar'):
        super(Langevin, self).__init__()
        if unet is None:
            unet, beta, img_shape = get_unet(mode=mode)
        self.device = torch.device('cuda')
        self.unet = unet
        self.mode = mode

    def sample(self, score_step_size=1e-3, noise_step_size=0, total_step=1000,
               reserve_process_img=True):
        x = torch.randn((1, 3, 32, 32), device=self.device)
        for step in range(1, total_step + 1):
            score = self.unet(x, )
