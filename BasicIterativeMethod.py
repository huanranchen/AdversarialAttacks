import torch
import torchvision.transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from utils import *


def get_fool_image(image: torch.tensor,
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

    x = image
    x.requires_grad = True
    false_label = 999
    optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for step in range(1, max_fool_time + 1):
        pre = model(x)
        _, now = torch.max(pre, dim=1)
        if now != category:
            print('-' * 100)
            print('managed to generate fool image')
            x = x.detach()
            show_image(x)
            return x
        else:
            loss = criterion(pre, torch.tensor([false_label]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('-'*100)
    print('failed to generate fool image')