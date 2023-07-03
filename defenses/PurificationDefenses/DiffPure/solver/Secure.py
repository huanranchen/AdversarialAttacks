import torch
from torch import nn, Tensor
from tqdm import tqdm
from attacks import AdversarialInputAttacker
from torchvision import transforms
import os
import numpy as np
from data import get_someset_loader


class SecureSolver():
    def __init__(self,
                 unet: nn.Module,
                 beta,
                 classifier: nn.Module,
                 device=torch.device('cuda'),
                 secure_scale=0.1,
                 T=1000):
        self.device = device
        self.unet = unet
        self.beta = beta
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.classifier = classifier
        self.secure_scale = secure_scale
        self.T = T

        # training schedule
        self.criterion = nn.CrossEntropyLoss()
        self.unet_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=4e-4)

        self.init()

    def init(self):
        # init
        self.classifier.eval().requires_grad_(False).to(self.device)
        self.unet.eval().requires_grad_(False).to(self.device)

    def get_grad(self, x: Tensor, y: Tensor) -> Tensor:
        x.requires_grad = True
        loss = self.criterion(self.classifier(x), y)
        loss.backward()
        grad = x.grad
        x.requires_grad = False
        return grad

    def train(self, train_loader, total_epoch=100):
        self.unet.train()
        self.unet.requires_grad_(True)
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                # some preprocess
                x = (x - 0.5) * 2
                # train
                x, y = x.to(self.device), y.to(self.device)
                t = torch.randint(1000, (x.shape[0],), device=self.device)
                tensor_t = t
                noise = torch.randn_like(x)
                noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x + \
                           torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
                pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
                target = noise - self.secure_scale * self.get_grad(x, y) * \
                         (torch.sqrt(self.alpha_bar[t]) / torch.sqrt(1 - self.alpha_bar[t])).view(-1, 1, 1, 1)
                loss = self.unet_criterion(pre, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            print(f'epoch {epoch}, loss {epoch_loss / len(train_loader)}')
            torch.save(self.unet.state_dict(), 'unet.pt')

        self.init()
