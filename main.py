from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, DiffusionAttacker, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, SGD
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt
from defenses import DiffusionPureImageNet
from torchvision import transforms
from PIL import Image
from tester import test_transfer_attack_acc, test_transfer_attack_acc_distributed


loader = get_NIPS17_loader(batch_size=1, shuffle=True, num_workers=0)
to_img = transforms.ToPILImage()

model = DiffusionPureImageNet().to(torch.device('cpu'))
diffusion = model
classifier = BaseNormModel(resnet50(pretrained=True)).to(torch.device('cpu'))
# attacker = DiffusionAttacker([diffusion])
attacker = MI_FGSM([classifier])

# for x, y in loader:
#     x, y = x[2].cuda().unsqueeze(0), y[2].cuda().unsqueeze(0)
#     x = F.interpolate(x, size=(256, 256))
#     # making adv_x
#     adv_x = x.clone()
#     adv_x = attacker(adv_x, y)
#     adv_img: Image.Image = to_img(adv_x.squeeze())
#     adv_img.save('adv.png')
#     # making purified image
#     purified_x = diffusion(adv_x)
#     purified_img: Image.Image = to_img(purified_x.squeeze())
#     purified_img.save('purified.png')
#     print(y, torch.max(classifier(adv_x), dim=1)[1], torch.max(classifier(purified_x), dim=1)[1])
#     break


class TargetModel(torch.nn.Module):
    def __init__(self):
        super(TargetModel, self).__init__()
        self.diffusion = diffusion
        self.classifier = classifier

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256))
        x = self.diffusion(x)
        x = self.classifier(x)
        return x


def get_attacker():
    return MI_FGSM([classifier])


def get_target_model():
    return [TargetModel()]


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    test_transfer_attack_acc_distributed(get_attacker, loader, get_target_model, num_gpu=4)
