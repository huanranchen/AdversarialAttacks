import torch
import torchvision.transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from utils import *


def perturb(x: torch.tensor, eps: int = 0.3) -> torch.tensor:
    return x + torch.rand(x.shape) * eps


def project(original_image: torch.tensor,
            now_image: torch.tensor,
            eps: int = 0.3, ) -> torch.tensor:
    now_image = torch.clip(now_image, min=original_image - eps, max=original_image + eps)
    return torch.clip(now_image, min=0, max=1)


def get_fool_image(image: torch.tensor,
                   total_attempt: int = 1,
                   max_fool_time: int = 9,
                   lr: int = 1e-3, ) -> Image:
    if len(image.shape) <= 3:
        image = image.unsqueeze(0)
    model = models.resnet18(pretrained=True)
    with torch.no_grad():
        x = model(image)
        x = F.softmax(x, dim=1)
        prob, category = torch.max(x, dim=1)
        print(f'now category is {category}, confidence is {prob}')


    for attempt in range(1, total_attempt + 1):
        x = image.clone()
        x = perturb(x)
        x.requires_grad = True
        optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9, maximize=True)
        criterion = nn.CrossEntropyLoss()
        max_loss = 0
        for step in range(1, max_fool_time + 1):
            pre = model(x)
            _, now = torch.max(pre, dim=1)
            if now != category:
                print('-' * 100)
                print(f'attempt {attempt}, managed to generate fool image, loss = {max_loss}')
                x = x.detach()
                show_image(x)
                break
            else:
                loss = criterion(pre, torch.tensor([category]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.item() > max_loss:
                    max_loss = loss.item()

        print('-' * 100)
        print(f'attempt {attempt}, failed to generate fool image')

if __name__ == '__main__':
    x = get_image()
    get_fool_image(x)

