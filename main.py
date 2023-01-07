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

loader = get_NIPS17_loader(batch_size=5, shuffle=True)
to_img = transforms.ToPILImage()

model = DiffusionPureImageNet()
diffusion = model
classifier = BaseNormModel(resnet50(pretrained=True)).cuda()
# attacker = DiffusionAttacker([diffusion])
attacker = MI_FGSM([classifier])
for x, y in loader:
    x, y = x[2].cuda().unsqueeze(0), y[2].cuda().unsqueeze(0)
    x = F.interpolate(x, size=(256, 256))
    # making adv_x
    adv_x = x.clone()
    adv_x = attacker(adv_x, y)
    adv_img: Image.Image = to_img(adv_x.squeeze())
    adv_img.save('adv.png')
    # making purified image
    purified_x = diffusion(adv_x)
    purified_img: Image.Image = to_img(purified_x.squeeze())
    purified_img.save('purified.png')
    print(y, torch.max(classifier(adv_x), dim=1)[1], torch.max(classifier(purified_x), dim=1)[1])
    break

