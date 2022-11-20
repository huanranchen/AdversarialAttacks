import os
from data import get_NIPS17_loader
from attacks import FGSM
from utils import scale_and_show_tensor
from models import BaseNormModel, resnet50
import torch

loader = get_NIPS17_loader()
model = resnet50(pretrained=True)
model = BaseNormModel(model)
attacker = FGSM(model)

count = 0
os.makedirs('./visualize_perturbation/')
for x, y in loader:
    attacked = attacker(x, y).detach()
    perturbation = attacked - x
    perturbation = torch.split(perturbation, 1, dim=1)
    for p in perturbation:
        p = scale_and_show_tensor(p)
        p.save(f'./visualize_perturbation/{count}')
        count += 1
