import os
from data import get_NIPS17_loader
from attacks import FGSM, PGD
from utils import scale_and_show_tensor
from models import BaseNormModel, resnet50
import torch
from tqdm import tqdm
from utils import total_variation
from torch.nn import functional as F


def criterion(x, y, perturbation, alpha=1, beta=100):
    cross_entropy = F.cross_entropy(x, y)
    tv = total_variation(perturbation)
    return alpha * cross_entropy + beta * tv


loader = get_NIPS17_loader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50(pretrained=True)
model = BaseNormModel(model).to(device)
model.eval()
attacker = PGD(model, total_step=100, criterion=criterion)

count = 0

path = './visualize_perturbation/PGD100tv100/'

if not os.path.exists(path):
    os.makedirs(path)
for x, y in tqdm(loader):
    x, y = x.to(device), y.to(device)
    attacked = attacker(x, y).detach()
    perturbation = attacked - x
    perturbation = torch.split(perturbation, 1, dim=0)
    for p in perturbation:
        p = scale_and_show_tensor(p)
        p.save(os.path.join(path, f'{count}.png'))
        count += 1
