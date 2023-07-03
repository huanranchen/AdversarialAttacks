import torch
from torch import nn, Tensor
from tqdm import tqdm
from attacks import AdversarialInputAttacker
from torchvision import transforms
import os
import numpy as np
from data import get_someset_loader


class AdversarialTrainingSolver():
    def __init__(self,
                 unet: nn.Module,
                 beta,
                 attacker: AdversarialInputAttacker,
                 device=torch.device('cuda'),
                 T=1000):
        self.device = device
        self.unet = unet
        self.beta = beta
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.attacker = attacker
        self.T = T

        # training schedule
        self.criterion = nn.CrossEntropyLoss()
        self.unet_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=4e-4)

        self.init()

    def init(self):
        # init
        self.unet.eval().requires_grad_(False).to(self.device)
        self.ground_truth = {}
        self.to_img = transforms.ToPILImage()
        self.img_saving_path = './resources/synthesized_dataset/cifar10/'

    @torch.no_grad()
    def save_batch(self, x, y, begin_id):
        '''

        :param x:  N, C, H, D
        :param y:  N
        :param begin_id: for naming the images. Don't care about that
        :return:
        '''
        x, y = x.cpu(), y.cpu()
        for i in range(y.shape[0]):
            now_x, now_y = x[i, :, :, :], y[i].item()
            self.ground_truth[str(begin_id + i) + '.jpg'] = now_y
            now_x = self.to_img(now_x)
            now_x.save(os.path.join(self.img_saving_path, str(begin_id + i) + '.jpg'))

    def set_generate(self, loader):
        count = 0
        for x, y in tqdm(loader):
            x, y = x.to(self.device), y.to(self.device)
            adv_x = self.attacker(x, y)
            target_x = torch.cat([x, adv_x], dim=2)  # N, C, 64, 32
            self.save_batch(target_x, y, count)
            count += y.shape[0]
        np.save(os.path.join(self.img_saving_path, 'gt.npy'), self.ground_truth)
        print(f'sir, we have generated backdoored images. total {count}')
        print('-' * 100)

    def load(self):
        self.unet.load_state_dict(torch.load('unet_advtrain.pt'))
        print('ZhengyiAdvSolver: using loaded net')

    def train(self, train_loader, total_epoch=100):
        self.init()
        # self.set_generate(train_loader)
        # for model in self.attacker.models:
        #     del model
        # del self.attacker
        train_loader = get_someset_loader(self.img_saving_path, os.path.join(self.img_saving_path, 'gt.npy'),
                                          batch_size=64)
        self.unet.train()
        self.unet.requires_grad_(True)
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                # some preprocess
                x, y = x.to(self.device), y.to(self.device)
                x, adv_x = torch.split(x, 32, dim=2)
                x = (x - 0.5) * 2
                adv_x = (adv_x - 0.5) * 2
                # train
                t = torch.randint(1000, (x.shape[0],), device=self.device)
                tensor_t = t
                noise = torch.randn_like(x)
                noised_x = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * adv_x + \
                           torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * noise
                pre = self.unet(noised_x, tensor_t)[:, :3, :, :]
                target = noise
                target += (adv_x - x) * \
                          (torch.sqrt(self.alpha_bar[100]) / torch.sqrt(1 - self.alpha_bar[100])
                           ).repeat(x.shape[0], 1, 1, 1)
                loss = self.unet_criterion(pre, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            print(f'epoch {epoch}, loss {epoch_loss / len(train_loader)}')
            torch.save(self.unet.state_dict(), 'unet_advtrain.pt')

        self.init()


class AdversarialTrainingDiffusionOfDiffPure():
    def __init__(self,
                 diffusion: nn.Module,
                 classifier: nn.Module,
                 attacker: AdversarialInputAttacker,
                 device=torch.device('cuda'),
                 ):
        # TODO: first figure out how to use gradient checkpoint to diffattack
        # FIXME: then use this technique to get the gradient with respect to diffusion
        raise NotImplementedError
        self.device = device
        self.diffusion = diffusion
        self.classifier = classifier
        self.attacker = attacker

        # training schedule
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=4e-4)

        self._init()

    def init(self):
        # init
        self.diffusion.requires_grad_(True).train()
        self.classifier.requires_grad_(False).eval()

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
