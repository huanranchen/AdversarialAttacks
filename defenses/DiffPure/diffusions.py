import torch
import torchsde
from .eval_sde_adv import parse_args_and_config, robustness_eval
from torch import nn
from models import resnet50


def get_unet():
    from .guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    model_config = model_and_diffusion_defaults()
    model_config.update(vars(config.model))
    print(f'model_config: {model_config}')
    model, _ = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f'./resources/checkpoints/'
                                     f'DiffPure/256x256_diffusion_uncond.pt', map_location='cpu'))

    if model_config['use_fp16']:
        model.convert_to_fp16()
    betas = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
    return model, betas


class DiffusionSde(nn.Module):
    unet, betas = get_unet()

    def __init__(self, unet: nn.Module = unet, beta=betas, T=1000):
        super(DiffusionSde, self).__init__()
        self.device = torch.device('cuda')
        self.unet = unet
        alpha = (1 - beta)
        self.alpha_bar = alpha.cumprod(dim=0).to(self.device)
        self.beta = beta * T
        self.T = T
        self.eval().to(self.device).requires_grad_(False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def convert(self, x):
        return (x + 1) * 0.5

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
        pre = self.unet(x.view(N, 3, 256, 256), tensor_t)[:, :3, :, :].view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - diffusion ** 2 * score
        return -drift

    def f(self, t: float, x):
        f = self.reverse_diffusion_forward(x, int(t * self.T), return_type='drift')
        assert f.shape == x.shape
        # print('f', torch.sum(f))
        return f

    def g(self, t: float, x):
        g = self.reverse_diffusion_forward(x, int(t * self.T), return_type='diffusion')
        # print('g', torch.sum(g))
        return g

    @torch.no_grad()
    def sample(self):
        import torchsde
        x = torch.randn((1, 3, 256, 256), device=self.device).view((1, 3 * 256 * 256))
        ts = torch.tensor([0., 1 - 1e-4], device=self.device)
        x = torchsde.sdeint_adjoint(self, x, ts, method='euler')
        x = x.reshape(2, 1, 3, 256, 256)
        return self.convert(x[1])

    def purify(self, x, noise_level=150):
        x = (x - 0.5) * 2
        x = torch.sqrt(self.alpha_bar[self.T - noise_level - 1]) * x + \
            torch.randn_like(x) * torch.sqrt(1 - self.alpha_bar[self.T - noise_level - 1])
        N = x.shape[0]
        x = x.view(N, -1)
        ts = torch.tensor([1 - noise_level * 1e-3, 1 - 1e-4], device=self.device)
        x = torchsde.sdeint_adjoint(self, x, ts, method='euler')
        x = x[-1]
        x = x.view(N, 3, 256, 256)
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
        pre = self.unet(x.view(N, 3, 256, 256), tensor_t)[:, :3, :, :].view(N, -1)
        score = - pre / torch.sqrt(1 - self.alpha_bar[self.T - t - 1])
        drift = forward_drift - 0.5 * diffusion ** 2 * score
        return -drift

    def f(self, t: float, x):
        f = self.reverse_diffusion_forward(x, int(t * self.T), return_type='drift')
        assert f.shape == x.shape
        # print('f', torch.sum(f))
        return f

    def g(self, t: float, x):
        return torch.zeros_like(x)


def ddpm(unet, device, betas, alpha, alpha_bar, ):
    x = torch.randn((1, 3, 256, 256), device=device)
    for t in range(999, 0, -1):
        tensor_t = torch.zeros((x.shape[0]), device=device) + t
        predict = unet(x, tensor_t)[:, :3, :, :]
        if t > 1:
            noise = torch.randn_like(x) * (
                    torch.sqrt(betas[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t]))
            # noise = torch.randn_like(x) * torch.sqrt(betas[t])
        else:
            noise = 0
        x = 1 / torch.sqrt(alpha[t]) * (x - (betas[t]) / torch.sqrt(1 - alpha_bar[t]) * predict) + noise
    return x


def ddim(unet, device, alpha_bar, stride=50):
    x = torch.randn((1, 3, 256, 256), device=device)
    alpha_bar = alpha_bar[::stride]
    embedding_t = list(range(1000))[::stride]
    for t in range(alpha_bar.shape[0] - 1, 0, -1):
        # sigma = torch.sqrt(1 - alpha[t]) * torch.sqrt(1 - alpha_bar[t - 1]) / torch.sqrt(1 - alpha_bar[t])
        sigma = 0
        tensor_t = torch.zeros((x.shape[0]), device=device) + embedding_t[t]
        predict = unet(x, tensor_t)[:, :3, :, :]
        x0 = (x - torch.sqrt(1 - alpha_bar[t]) * predict) / torch.sqrt(alpha_bar[t])
        if t > 1:
            noise = torch.randn_like(x) * sigma
        else:
            noise = 0
        x = torch.sqrt(alpha_bar[t - 1]) * x0 + torch.sqrt(1 - alpha_bar[t - 1] - sigma ** 2) * predict + noise
    return x


class DiffusionPureImageNet(nn.Module):
    def __init__(self):
        super(DiffusionPureImageNet, self).__init__()
        self.device = torch.device('cuda')
        self.diffusion = DiffusionSde()
        self.model = resnet50(pretrained=True)
        self.eval().requires_grad_(False)
        self.to(self.device)

    def forward(self, x, *args, **kwargs):
        x = self.diffusion(x)
        x = self.model(x)
        return x
