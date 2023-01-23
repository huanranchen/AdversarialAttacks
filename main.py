import torch
from attacks import MI_FGSM
from data import get_NIPS17_loader
from tester import test_autoattack_acc, test_transfer_attack_acc
from defenses import DiffusionPureImageNet
from torchvision import transforms
import numpy as np
import random

# torch.manual_seed(1)
# random.seed(1)
# np.random.seed(1)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_NIPS17_loader(batch_size=1)


def save_img(x):
    if x.ndim == 4:
        x = x[0]
    img = to_img(x)
    img.save('test.png')


with torch.no_grad():
    diffpure = DiffusionPureImageNet()
    diffusion = diffpure.diffusion
    diffusion.sample()
#     x = next(iter(loader))[0].cuda()
#     diffusion.purify(x)

# test_transfer_attack_acc(MI_FGSM([diffpure]), loader, [diffpure])
