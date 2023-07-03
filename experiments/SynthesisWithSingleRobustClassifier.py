import torch
from torchvision import transforms
import numpy as np
import random
from models.RobustBench import *
from torch.nn import functional as F
from data import get_CIFAR10_test

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

loader = get_CIFAR10_test(batch_size=1)
for x, y in loader:
    if y.item() == 1:
        break

device = torch.device('cuda')
target = torch.tensor([0], device=device)
model = Rice2020OverfittingNetL2(pretrained=True).cuda().requires_grad_(False).eval()
# x = torch.randn((1, 3, 32, 32), device=device)
x = x.cuda()
original_x = x.clone()
criterion = lambda x: F.cross_entropy(x, target)
joint_distribution_criterion = lambda x: -torch.sum(x[:, target])
# step_size = 16 / 255 / 160
step_size = 0.1
to_img = transforms.ToPILImage()
origin = to_img(x.squeeze())
origin.save("ori_img.png")
epsilon = 0.5
for i in range(7):
    x.requires_grad = True
    pre = model(x)
    loss = joint_distribution_criterion(pre)
    loss.backward()
    grad = x.grad.clone()
    x.requires_grad = False
    x -= step_size * grad.sign()
    x = torch.clamp(x, min=0, max=1)
    # x = torch.clamp(x, min=original_x - epsilon, max=original_x + epsilon)

img = to_img(x.squeeze())
img.save('./robust_classifier_synthesis.png')
