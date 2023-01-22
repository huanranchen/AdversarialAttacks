from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, DiffusionAttacker, DiffusionPatchAttacker, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, SGD, DI_MI_FGSM
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt
from defenses import DiffusionPureImageNet
from torchvision import transforms
from PIL import Image
from tester import test_transfer_attack_acc, test_transfer_attack_acc_distributed

to_img = transforms.ToPILImage()


def show_img(x: torch.Tensor):
    x: torch.Tensor = (x + 1) * 0.5
    x = torch.clamp(x, min=0, max=1)
    if len(x.shape) == 4:
        x = x[0]
    x: Image.Image = to_img(x)
    x.save('test.png')


device = torch.device('cuda')
to_tensor = transforms.ToTensor()
classifier = BaseNormModel(resnet50(pretrained=True)).cuda()
classifier.eval()
classifier.requires_grad_(False)
model = DiffusionPureImageNet()
diffusion = model.cuda()
diffusion.eval()
diffusion.requires_grad_(False)

runner = diffusion.model
unet = runner.model.to(device).eval()
betas = runner.betas
alpha = (1 - betas)
alpha_bar = alpha.cumprod(dim=0).to(device)


def ddpm():
    x = torch.randn((1, 3, 256, 256), device=device)
    for t in range(999, -1, -1):
        tensor_t = torch.zeros((x.shape[0]), device=device) + t
        predict = unet(x, tensor_t)[:, :3, :, :]
        if t > 1:
            noise = torch.randn_like(x) * (
                    torch.sqrt(betas[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t]))
            # noise = torch.randn_like(x) * torch.sqrt(betas[t])
        else:
            noise = 0
        x = 1 / torch.sqrt(alpha[t]) * (x - (betas[t]) / torch.sqrt(1 - alpha_bar[t]) * predict) + noise

    show_img(x)


def ddim():
    x = torch.randn((1, 3, 256, 256), device=device)
    for t in range(999, -1, -1):
        sigma = torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t])
        tensor_t = torch.zeros((x.shape[0]), device=device) + t
        predict = unet(x, tensor_t)[:, :3, :, :]
        x0 = (x - torch.sqrt(1 - alpha_bar[t]) * predict) / torch.sqrt(alpha_bar[t])
        if t > 1:
            noise = torch.randn_like(x) * sigma
        else:
            noise = 0
        x = torch.sqrt(alpha_bar[t - 1]) * x0 + torch.sqrt(1 - alpha_bar[t - 1] - sigma ** 2) * predict + noise

    show_img(x)


with torch.no_grad():
    ddim()
