import torch
from attacks import MI_FGSM
from data import get_NIPS17_loader, get_CIFAR10_train
from tester import test_autoattack_acc, test_transfer_attack_acc
from defenses import DiffusionPureImageNet
from torchvision import transforms
import numpy as np
import random
from defenses.DiffPure import SecureSolver


torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)


to_img = transforms.ToPILImage()
loader = get_CIFAR10_train(batch_size=128)


def save_img(x):
    if x.ndim == 4:
        x = x[0]
    img = to_img(x)
    img.save('test.png')


with torch.no_grad():
    diffpure = DiffusionPureImageNet(mode='sde',
                                     # post_transforms=transforms.Compose(
                                     #     [
                                     #         transforms.Resize((224, 224)),
                                     #
                                     #     ]
                                     # )
                                     )
    solver = SecureSolver(diffpure.diffusion.unet,
                          torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda')),
                          diffpure.model)
    # for x, y in loader:
    #     x = diffpure(x.cuda())
    #     break
    # diffusion = diffpure.diffusion
#     diffusion.sample()
#     x = next(iter(loader))[0].cuda()
#     diffusion.purify(x)
#
# test_transfer_attack_acc(MI_FGSM([diffpure], epsilon=4/255), loader, [diffpure])
solver.train(loader)
