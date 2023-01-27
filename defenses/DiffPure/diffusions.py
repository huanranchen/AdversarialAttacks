import torch
import torchsde
from .eval_sde_adv import parse_args_and_config_imagenet, robustness_eval, parse_args_and_config_cifar
from torch import nn
from models import resnet50
from torchvision import transforms
from .runners.diffpure_sde import RevVPSDE, restore_checkpoint
from .score_sde.models import utils as mutils
from .guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from .score_sde.losses import get_optimizer
from .score_sde.models import utils as mutils
from .score_sde.models.ema import ExponentialMovingAverage
from .score_sde import sde_lib


def get_unet(mode='cifar'):
    if mode == 'imagenet':
        args, config = parse_args_and_config_imagenet()
    elif mode == 'cifar':
        args, config = parse_args_and_config_cifar()
    if mode == 'imagenet':
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(config.model))
        print(f'model_config: {model_config}')
        model, _ = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(f'./resources/checkpoints/'
                                         f'DiffPure/256x256_diffusion_uncond.pt', map_location='cpu'))
        if model_config['use_fp16']:
            model.convert_to_fp16()
        img_shape = (3, 256, 256)

    if mode == 'cifar':
        img_shape = (3, 32, 32)
        print(f'model_config: {config}')
        model = mutils.create_model(config)
        optimizer = get_optimizer(config, model.parameters())
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        state = dict(step=0, optimizer=optimizer, model=model, ema=ema)
        restore_checkpoint(f'./resources/checkpoints/'
                           f'DiffPure/32x32_diffusion.pth',
                           state, torch.device('cuda'))
        ema.copy_to(model.parameters())
    betas = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
    return model, betas, img_shape


class DiffusionSde(nn.Module):
    def __init__(self, unet: nn.Module = None, beta=None, img_shape=None, T=1000, dt=1e-3):
        super(DiffusionSde, self).__init__()
        if unet is None:
            unet, beta, img_shape = get_unet()
        if beta is None:
            beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
        self.device = torch.device('cuda')
        self.unet = unet
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.beta = beta * T
        self.T = T
        self.eval().to(self.device).requires_grad_(False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.to_img = transforms.ToPILImage()
        self.i = 0
        self.dt = dt
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]

    def convert(self, x):
        x = (x + 1) * 0.5
        img = self.to_img(x[0])
        img.save(f'./what/{self.i}.png')
        self.i += 1
        return x

    def diffusion_forward(self, x: torch.Tensor, t: int):
        assert len(x.shape) == 2, 'x should be N, D'
        N = x.shape[0]
        diffusion = torch.sqrt(self.beta[t]).view(1, 1).repeat(N, x.shape[1])
        drift = -0.5 * self.beta[t] * x
        return drift, diffusion

    def reverse_diffusion_forward(self, x: torch.int, t: int, return_type='diffusion'):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.int) + (self.T - t - 1)
        forward_drift, forward_diffusion = self.diffusion_forward(x, self.T - t - 1)
        diffusion = forward_diffusion
        if return_type == 'diffusion':
            return diffusion
        pre = self.unet(x.view(N, *self.img_shape), tensor_t)[:, :3, :, :].view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - diffusion ** 2 * score
        return -drift

    def f(self, t: float, x):
        f = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='drift')
        assert f.shape == x.shape
        return f

    def g(self, t: float, x):
        g = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='diffusion')
        return g

    @torch.no_grad()
    def sample(self):
        import torchsde
        x = torch.randn((1, *self.img_shape), device=self.device).view((1, self.state_size))
        ts = torch.tensor([0., 1. - 1e-4], device=self.device)
        x = torchsde.sdeint(self, x, ts, method='euler')
        x = x[-1]  # N, 3, 256, 256
        x = x.reshape(1, *self.img_shape)
        return self.convert(x)

    def purify(self, x, noise_level=100):
        x = (x - 0.5) * 2
        x = torch.sqrt(self.alpha_bar[noise_level - 1]) * x + \
            torch.randn_like(x, requires_grad=False) * torch.sqrt(1 - self.alpha_bar[noise_level - 1])
        N = x.shape[0]
        x = x.view(N, -1)
        ts = torch.linspace(1 - noise_level * 1e-3, 1 - 1e-4, 2)
        x = torchsde.sdeint_adjoint(self, x, ts, method='euler', dt=self.dt)
        x = x.view(ts.shape[0], N, *self.img_shape)
        x = x[-1]
        x = x.view(N, *self.img_shape)
        return self.convert(x)

    def __call__(self, *args, **kwargs):
        return self.purify(*args, **kwargs)


class DiffusionOde(DiffusionSde):
    def __init__(self, *args, **kwargs):
        super(DiffusionOde, self).__init__(*args, **kwargs)

    def reverse_diffusion_forward(self, x: torch.int, t: int, return_type='diffusion'):
        N = x.shape[0]
        tensor_t = torch.zeros((N,), device=self.device, dtype=torch.int) + (self.T - t - 1)
        forward_drift, forward_diffusion = self.diffusion_forward(x, self.T - t - 1)
        diffusion = forward_diffusion
        if return_type == 'diffusion':
            return diffusion
        pre = self.unet(x.view(N, *self.img_shape), tensor_t)[:, :3, :, :].view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - 0.5 * diffusion ** 2 * score
        return -drift

    def f(self, t: torch.tensor, x):
        f = self.reverse_diffusion_forward(x, round(float(t) * self.T), return_type='drift')
        assert f.shape == x.shape
        # print('f', torch.sum(f))
        return f

    def g(self, t: float, x):
        return torch.zeros_like(x)


#
# def ddpm(unet, device, betas, alpha, alpha_bar, ):
#     x = torch.randn((1, 3, 256, 256), device=device)
#     for t in range(999, 0, -1):
#         tensor_t = torch.zeros((x.shape[0]), device=device) + t
#         predict = unet(x, tensor_t)[:, :3, :, :]
#         if t > 1:
#             noise = torch.randn_like(x) * (
#                     torch.sqrt(betas[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t]))
#             # noise = torch.randn_like(x) * torch.sqrt(betas[t])
#         else:
#             noise = 0
#         x = 1 / torch.sqrt(alpha[t]) * (x - (betas[t]) / torch.sqrt(1 - alpha_bar[t]) * predict) + noise
#     return x


class DDIM(nn.Module):
    def __init__(self, unet=None, beta=None, img_shape=None, T=1000, stride=1):
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
            x = torch.sqrt(alpha_bar[t - 1]) * x0 + torch.sqrt(1 - alpha_bar[t - 1] - sigma ** 2) * predict + noise
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


class DiffusionPureImageNet(nn.Module):
    def __init__(self, mode='sde',
                 pre_transforms=nn.Identity(),
                 post_transforms=nn.Identity(),
                 *args, **kwargs):
        super(DiffusionPureImageNet, self).__init__()
        self.device = torch.device('cuda')
        if mode == 'sde':
            self.diffusion = DiffusionSde(*args, **kwargs)
        elif mode == 'ode':
            self.diffusion = DiffusionOde(*args, **kwargs)
        elif mode == 'ddim':
            self.diffusion = DDIM(*args, **kwargs)
        self.model = resnet50(pretrained=True)
        self.eval().requires_grad_(False)
        self.to(self.device)
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

    def forward(self, x, *args, **kwargs):
        x = self.pre_transforms(x)
        x = self.diffusion(x)
        x = self.post_transforms(x)
        x = self.model(x)
        return x
