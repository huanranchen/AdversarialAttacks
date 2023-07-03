import torch
from torch import nn, Tensor
from tqdm import tqdm
from attacks import AdversarialInputAttacker
from torchvision import transforms
import random


class ConditionSolver():
    def __init__(self,
                 unet: nn.Module,
                 beta=torch.linspace(0.1 / 1000, 20 / 1000, 1000),
                 attacker: AdversarialInputAttacker or None = None,
                 device=torch.device('cuda'),
                 T=1000):
        self.device = device
        self.unet = unet
        self.beta = beta
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.T = T

        # training schedule
        self.criterion = nn.CrossEntropyLoss()
        self.unet_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        self.attacker = attacker

        self.init()
        self.transform = lambda x: (x - 0.5) * 2

    def init(self):
        # init
        self.unet.eval().requires_grad_(False).to(self.device)

    def train(self, train_loader, total_epoch=10000, p_uncondition=0.1):
        self.unet.train()
        self.unet.requires_grad_(True)
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.cuda(), y.cuda()
                if self.attacker is not None:
                    adv_x = self.attacker(x, y)
                    x, y = torch.cat([x, adv_x], dim=0), torch.cat([y, y], dim=0)
                # some preprocess
                x = self.transform(x)
                # train
                x, y = x.to(self.device), y.to(self.device)
                t = torch.randint(self.T, (x.shape[0],), device=self.device)
                tensor_t = t
                noise = torch.randn_like(x)
                noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x + \
                           torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
                if random.random() < p_uncondition:
                    pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
                else:
                    pre = self.unet(noised_x, tensor_t, y)[:, :3, :, :]
                target = noise
                loss = self.unet_criterion(pre, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            print(f'epoch {epoch}, loss {epoch_loss / len(train_loader)}')
            torch.save(self.unet.state_dict(), 'unet_horizontalflip_continue.pt')
            # if epoch % 100 == 0:
            #     self.optimizer.param_groups[0]['lr'] *= 0.9

        self.init()
