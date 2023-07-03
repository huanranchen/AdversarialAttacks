import torch
from torch import nn, Tensor
from tqdm import tqdm
from attacks import AdversarialInputAttacker
from torchvision import transforms
import os
import numpy as np
from data import get_someset_loader


class AdversarialTrainingClassifierOfDiffPure():
    def __init__(self,
                 diffusion: nn.Module,
                 classifier: nn.Module,
                 attacker: AdversarialInputAttacker,
                 device=torch.device('cuda'),
                 ):
        self.device = device
        self.diffusion = diffusion
        self.classifier = classifier
        self.attacker = attacker

        # training schedule
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=4e-4)

        self.init()

    def init(self):
        # init
        self.diffusion.eval().requires_grad_(False)
        self.classifier.requires_grad_(True).train()

    def train(self, train_loader, total_epoch=100):
        self.init()
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                # some preprocess
                x, y = x.to(self.device), y.to(self.device)
                adv_x = self.attacker(x, y)
                x, adv_x = (x - 0.5) * 2, (adv_x - 0.5) * 2
                # train
                self.classifier.train()
                purified = self.diffusion(adv_x)
                pre = self.classifier(purified)
                loss = self.criterion(pre, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            print(f'epoch {epoch}, loss {epoch_loss / len(train_loader)}')
            torch.save(self.classifier.state_dict(), 'adv_wideresnet.pt')

        self.init()
