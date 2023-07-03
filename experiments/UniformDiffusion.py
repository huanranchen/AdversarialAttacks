import torch
from attacks import BIM
from data import get_CIFAR10_test
from tester import test_transfer_attack_acc, test_acc
from torchvision import transforms
import numpy as np
import random
from defenses.PurificationDefenses.UniformDiffusionPure import UniformDiffusionSampler, UniformDiffusionSolver
from models import WideResNet_70_16_dropout
from torch.utils.checkpoint import checkpoint


torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=16)
device = torch.device('cuda')


def save_img(x, name='test'):
    if x.ndim == 4:
        x = x[0]
    img = to_img(x)
    img.save(f'{name}.png')


model = WideResNet_70_16_dropout()


class DiffPure(torch.nn.Module):
    def __init__(self, diffusion, model):
        super(DiffPure, self).__init__()
        self.diffusion = diffusion
        self.model = model

    def forward(self, x):
        x = self.diffusion(x)
        x = self.model(x)
        return x


class SubstituteUnet(torch.nn.Module):
    def __init__(self, unet):
        super(SubstituteUnet, self).__init__()
        self.unet = unet

    def forward(self, *args, **kwargs):
        x = checkpoint(self.unet, *args, **kwargs)
        return x


with torch.no_grad():
    diffusion = UniformDiffusionSampler()
    # diffpure = KnnDiffusionClassifier()
    diffusion.unet.load_state_dict(torch.load('./uniform_diffusion_unet.pt'))
    diffusion.unet = SubstituteUnet(diffusion.unet)

diffpure = DiffPure(diffusion, model)
diffpure.eval().requires_grad_(False).to(device)
# from torch import fx
# traced = fx.symbolic_trace(unet)
# print(traced.code)
# unet.load_state_dict(torch.load('./unet_advtrain.pt'))

# attacker = BIM(
#     [
#         diffpure,
#     ],
#     epsilon=8 / 255,
#     step_size=8 / 255 / 10,
# )
# train_loader = get_CIFAR10_train(batch_size=1)
# solver = ConditionSolver(unet, beta, attacker)
# solver.train(train_loader, total_epoch=200)

test_acc(diffpure, loader)
# I-FGSM
test_transfer_attack_acc(BIM([diffpure],
                             epsilon=8 / 255,
                             step_size=8 / 255 / 10, total_step=10),
                         loader, [diffpure])

# test_apgd_dlr_acc(diffpure, loader=loader, bs=1)

# #diffattack
# test_transfer_attack_acc(
#     DiffAttack([diffpure], total_step=100),
#     loader,
#     [diffpure],
# )
