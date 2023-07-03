import torch
from torch import nn
from torchvision import transforms
from defenses.PurificationDefenses.DiffPure.diffusions.model import get_unet


class DDIM(nn.Module):
    def __init__(self, unet=None, beta=None, img_shape=(3, 32, 32), T=1000, stride=1, ):
        super(DDIM, self).__init__()
        if unet is None:
            unet, beta, img_shape = get_unet()
        if beta is None:
            beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
        self.device = torch.device('cuda')
        self.unet = unet
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.beta = beta
        self.T = T
        self.eval().to(self.device).requires_grad_(False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.to_img = transforms.ToPILImage()
        self.i = 0
        self.stride = stride
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]

    def convert(self, x):
        x = (x + 1) * 0.5
        img = self.to_img(x[0])
        img.save(f'./what/{self.i}.png')
        self.i += 1
        return x

    def sample(self, stride=1):
        x = torch.randn((1, *self.img_shape), device=self.device)
        alpha_bar = self.alpha_bar[::stride]
        embedding_t = list(range(1000))[::stride]
        for t in range(alpha_bar.shape[0] - 1, 0, -1):
            # sigma = torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t])
            sigma = 0
            tensor_t = torch.zeros((x.shape[0]), device=self.device) + embedding_t[t]
            predict = self.unet(x, tensor_t)[:, :3, :, :]
            x0 = (x - torch.sqrt(1 - alpha_bar[t]) * predict) / torch.sqrt(alpha_bar[t])
            if t > 1:
                noise = torch.randn_like(x) * sigma
            else:
                noise = 0
            x = torch.sqrt(alpha_bar[t - 1]) * x0 + \
                torch.sqrt(1 - alpha_bar[t - 1] - sigma ** 2) * predict + \
                noise
        return self.convert(x)

    def purify(self, x, noise_level=150):
        x = (x - 0.5) * 2
        x = torch.sqrt(self.alpha_bar[noise_level - 1]) * x + \
            torch.randn_like(x, requires_grad=False) * torch.sqrt(1 - self.alpha_bar[noise_level - 1])
        alpha_bar = self.alpha_bar[:noise_level][::self.stride]
        embedding_t = list(range(self.T))[:noise_level][::self.stride]
        for t in range(alpha_bar.shape[0] - 1, 0, -1):
            # sigma = torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t])
            sigma = 0
            tensor_t = torch.zeros((x.shape[0]), device=self.device) + embedding_t[t]
            predict = self.unet(x, tensor_t)[:, :3, :, :]
            x0 = (x - torch.sqrt(1 - alpha_bar[t]) * predict) / torch.sqrt(alpha_bar[t])
            if t > 1:
                noise = torch.randn_like(x) * sigma
            else:
                noise = 0
            x = torch.sqrt(alpha_bar[t - 1]) * x0 + torch.sqrt(1 - alpha_bar[t - 1] - sigma ** 2) * predict + noise
        x = self.convert(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.purify(*args, **kwargs)


#
def ddpm(unet, betas, y=None, w=10, device=torch.device('cuda')):
    y = torch.tensor([1], device=device)
    alpha = (1 - betas)
    alpha_bar = alpha.cumprod(dim=0).to(device)
    x = torch.randn((1, 3, 32, 32), device=device)
    for t in range(999, 1, -1):
        tensor_t = torch.zeros((x.shape[0]), device=device) + t
        pre_with_y = unet(x, tensor_t, y)[:, :3, :, :]
        pre_without_y = unet(x, tensor_t)[:, :3, :, :]
        predict = (w + 1) * pre_with_y - w * pre_without_y
        if t > 1:
            noise = torch.randn_like(x) * (
                    torch.sqrt(betas[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t]))
            # noise = torch.randn_like(x) * torch.sqrt(betas[t])
        else:
            noise = 0
        x = 1 / torch.sqrt(alpha[t]) * (x - (betas[t]) / torch.sqrt(1 - alpha_bar[t]) * predict) + noise
    return (x + 1) * 0.5
