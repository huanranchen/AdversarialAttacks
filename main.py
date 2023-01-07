from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, DiffusionAttacker, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, SGD
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt
from defenses import get_diffpure_imagenet
from torchvision import transforms
from PIL import Image

loader = get_NIPS17_loader(batch_size=1, shuffle=True)
to_img = transforms.ToPILImage()

model = get_diffpure_imagenet()
diffusion = model.runner
classifier = BaseNormModel(resnet50(pretrained=True))
attacker = DiffusionAttacker([diffusion])
x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
x = F.interpolate(x, size=(256, 256))
# normalize to diffusion input
adv_x = (x - 0.5) * 2
with torch.autograd.detect_anomaly():
    adv_x = attacker(adv_x, y)
adv_img: Image.Image = to_img((adv_x.squeeze()+1)*0.5)
adv_img.show()
purified_x = diffusion.image_editing_sample(adv_x)
purified_x = (purified_x + 1) * 0.5
purified_img: Image.Image = to_img(purified_x.squeeze())
purified_img.show()
print(y, torch.max(classifier(adv_x.cuda()), dim=1)[1], torch.max(classifier(purified_x.cuda()), dim=1)[1])
