import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
from torchvision import transforms
import random


class DiffusionAttacker(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 epsilon: float = 16 / 255,
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 0,
                 ):
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(DiffusionAttacker, self).__init__(model)
        self.to_img = transforms.ToPILImage()
        self.record_count = 0

    def attack(self, *args, **kwargs):
        return self.direct_attack(*args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    # def attack_first_iter(self, x, y, ):
    #     N = x.shape[0]
    #     original_x = x.clone()
    #     # momentum = torch.zeros_like(x)
    #     target = torch.zeros_like(x)
    #     if self.random_start:
    #         x = self.perturb(x)
    #
    #     total_loss = 0
    #     optimizer = torch.optim.Adam([x], lr=1e-3, amsgrad=True)
    #     for step in range(1, self.total_step):
    #         x.requires_grad = True
    #         out = 0
    #         for model in self.models:
    #             out += model(x, diffusion_iter_time=1, tag='sde_adv').to(x.device)
    #         out /= self.n
    #         # self.record(out, keyword='after')
    #         loss = self.criterion(out, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         # grad = x.grad
    #         # x.requires_grad = False
    #         # # update
    #         # if self.targerted_attack:
    #         #     momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #         #     x += self.step_size * momentum.sign()
    #         # else:
    #         #     momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #         #     x += self.step_size * momentum.sign()
    #         # x = clamp(x)
    #         # x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
    #         with torch.no_grad():
    #             x = x.clamp_(min=0, max=1)
    #             x = x.clamp_(min=original_x - self.epsilon, max=original_x + self.epsilon)
    #         if step % 10 == 0:
    #             print(f'step {step}, loss {total_loss / step}')
    #
    #     del optimizer
    #     del original_x
    #
    #     return x

    # def attack_training_objective(self, x, y, accumulation_step=100):
    #     N = x.shape[0]
    #     original_x = x.clone()
    #     # momentum = torch.zeros_like(x)
    #     if self.random_start:
    #         x = self.perturb(x)
    #
    #     total_loss = 0
    #     optimizer = torch.optim.Adam([x], lr=1e-1, maximize=True)
    #     for step in range(1, self.total_step):
    #         x.requires_grad = True
    #         optimizer.zero_grad()
    #         total_loss = 0
    #         for _ in range(accumulation_step):
    #             out, target = self.unet_forward(x)
    #             # self.record(out, keyword='after')
    #             loss = self.criterion(out, target)
    #             # print(out, target)
    #             # assert False
    #             loss *= 1e6
    #             loss.backward()
    #             total_loss += loss.item()
    #         x.grad /= accumulation_step
    #         # print(x.grad)
    #         total_loss /= accumulation_step
    #         optimizer.step()
    #         # total_loss += loss.item()
    #         # grad = x.grad
    #         print(total_loss)
    #         # print(grad)
    #         x.requires_grad = False
    #         # update
    #         # if self.targerted_attack:
    #         #     momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #         #     x += self.step_size * momentum.sign()
    #         # else:
    #         #     momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #         #     x += self.step_size * momentum.sign()
    #         # x = clamp(x)
    #         # x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
    #         with torch.no_grad():
    #             x = x.clamp_(min=0, max=1)
    #             x = x.clamp_(min=original_x - self.epsilon, max=original_x + self.epsilon)
    #         # if step % 10 == 0:
    #         #     print(f'step {step}, loss {total_loss / step}')
    #     del original_x
    #     return x

    def direct_attack(self, x, y, accumulation_step=1):
        original_x = x.clone()
        if self.random_start:
            x = self.perturb(x)
        optimizer = torch.optim.Adam([x], lr=1e-1, maximize=True)
        for step in range(1, self.total_step+1):
            x.requires_grad = True
            optimizer.zero_grad()
            total_loss = 0
            for _ in range(accumulation_step):
                out = self.models[0].forward(x)
                # self.record(out, keyword='after')
                loss = self.criterion(out, y)
                # # print(out, target)
                # # assert False
                # loss *= 1e6
                loss.backward()
                total_loss += loss.item()
            x.grad /= accumulation_step
            x.grad.clamp_(min=-1, max=1)
            # print(x.grad)
            total_loss /= accumulation_step
            optimizer.step()
            grad = x.grad
            print(total_loss)
            x.requires_grad = False
            # update
            # if self.targerted_attack:
            #     momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
            #     x += self.step_size * momentum.sign()
            # else:
            #     momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
            #     x += self.step_size * momentum.sign()
            # x = clamp(x)
            # x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            with torch.no_grad():
                x = x.clamp_(min=0, max=1)
                x = x.clamp_(min=original_x - self.epsilon, max=original_x + self.epsilon)
            # if step % 10 == 0:
            #     print(f'step {step}, loss {total_loss / step}')
        del original_x
        return x

    def record(self, x, keyword='before'):
        x = self.to_img(x[0])
        x.save(f'{keyword}{self.record_count}.png')
        self.record_count += 1

    def unet_forward(self, x, repeat_time=1):
        """
        缂轰釜楂樻柉鍣０
        :param x:
        :return:
        """
        x = (x - 0.5) * 2
        x = x.repeat(repeat_time, 1, 1, 1)
        out = 0
        N = x.shape[0]
        for model in self.models:
            e = torch.randn_like(x).to(self.device)
            e.requires_grad = False
            betas = model.model.betas
            a = (1 - betas).cumprod(dim=0).to(self.device)
            total_noise_levels = random.randint(1000 - 150, 1000)
            noise = e * (1.0 - a[total_noise_levels - 1]).sqrt()
            input = x * a[total_noise_levels - 1].sqrt() + noise
            t = total_noise_levels + torch.zeros((N,), device=self.device)
            out += model.model.model(input, t)  # UNet
        out /= self.n
        x = out
        x = x[:, :3, :, :]
        x = (x + 1) * 0.5
        return x, (e + 1) * 0.5


class DiffusionPatchAttacker(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 epsilon: float = 16 / 255,
                 total_step: int = 100, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.MSELoss(),
                 targeted_attack=False,
                 mu: float = 0,
                 ):
        self.random_start = random_start
        self.epsilon = epsilon
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(DiffusionPatchAttacker, self).__init__(model)
        self.to_img = transforms.ToPILImage()
        self.record_count = 0

    def attack(self, *args, **kwargs):
        return self.attack_training_objective(*args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    # def attack_first_iter(self, x, y, ):
    #     N = x.shape[0]
    #     original_x = x.clone()
    #     # momentum = torch.zeros_like(x)
    #     target = torch.zeros_like(x)
    #     if self.random_start:
    #         x = self.perturb(x)
    #
    #     total_loss = 0
    #     optimizer = torch.optim.Adam([x], lr=1e-3, amsgrad=True)
    #     for step in range(1, self.total_step):
    #         x.requires_grad = True
    #         out = 0
    #         for model in self.models:
    #             out += model(x, diffusion_iter_time=1, tag='sde_adv').to(x.device)
    #         out /= self.n
    #         # self.record(out, keyword='after')
    #         loss = self.criterion(out, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         # grad = x.grad
    #         # x.requires_grad = False
    #         # # update
    #         # if self.targerted_attack:
    #         #     momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #         #     x += self.step_size * momentum.sign()
    #         # else:
    #         #     momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #         #     x += self.step_size * momentum.sign()
    #         # x = clamp(x)
    #         # x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
    #         with torch.no_grad():
    #             x = x.clamp_(min=0, max=1)
    #             x = x.clamp_(min=original_x - self.epsilon, max=original_x + self.epsilon)
    #         if step % 10 == 0:
    #             print(f'step {step}, loss {total_loss / step}')
    #
    #     del optimizer
    #     del original_x
    #
    #     return x

    def attack_training_objective(self, x, y, accumulation_step=100):
        N = x.shape[0]
        original_x = x.clone()
        patch = torch.zeros((3, 32, 32), device=self.device, requires_grad=True)
        # momentum = torch.zeros_like(x)

        optimizer = torch.optim.Adam([patch], lr=4e-1, maximize=True)
        for step in range(1, self.total_step):
            optimizer.zero_grad()
            total_loss = 0
            for _ in range(accumulation_step):
                x = original_x.clone()
                x[:, :, :32, :32] = patch
                out, target = self.unet_forward(x)
                # self.record(out, keyword='after')
                loss = self.criterion(out, target)
                # print(out, target)
                # assert False
                loss *= 1e6
                loss.backward()
                total_loss += loss.item()
            patch.grad /= accumulation_step
            # print(x.grad)
            total_loss /= accumulation_step
            optimizer.step()
            # total_loss += loss.item()
            print(total_loss)
            # update
            # if self.targerted_attack:
            #     momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
            #     x += self.step_size * momentum.sign()
            # else:
            #     momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
            #     x += self.step_size * momentum.sign()
            with torch.no_grad():
                patch = patch.clamp_(min=0, max=1)
            if step % 10 == 0:
                print(f'step {step}, loss {total_loss}')

        return x.detach()

    def record(self, x, keyword='before'):
        x = self.to_img(x[0])
        x.save(f'{keyword}{self.record_count}.png')
        self.record_count += 1

    def unet_forward(self, x, repeat_time=1):
        """
        缂轰釜楂樻柉鍣０
        :param x:
        :return:
        """
        x = (x - 0.5) * 2
        x = x.repeat(repeat_time, 1, 1, 1)
        out = 0
        N = x.shape[0]
        for model in self.models:
            e = torch.randn_like(x).to(self.device)
            e.requires_grad = False
            betas = model.model.betas
            a = (1 - betas).cumprod(dim=0).to(self.device)
            total_noise_levels = random.randint(1000 - 150, 1000 - 149)
            noise = e * (1.0 - a[total_noise_levels - 1]).sqrt()
            input = x * a[total_noise_levels - 1].sqrt() + noise
            t = total_noise_levels + torch.zeros((N,), device=self.device)
            out += model.model.model(input, t)  # UNet
        out /= self.n
        x = out
        x = x[:, :3, :, :]
        x = (x + 1) * 0.5
        return x, (e + 1) * 0.5
