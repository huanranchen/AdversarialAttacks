from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, DiffusionAttacker, DiffusionPatchAttacker, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, SGD, DI_MI_FGSM
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt
from defenses import DiffusionPure
from torchvision import transforms
from PIL import Image
from tester import test_transfer_attack_acc, test_transfer_attack_acc_distributed

loader = get_NIPS17_loader(batch_size=1, shuffle=True, num_workers=0)
to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
classifier = BaseNormModel(resnet50(pretrained=True)).cuda()
classifier.eval()
classifier.requires_grad_(False)
model = DiffusionPure()
diffusion = model.cuda()
diffusion.eval()
diffusion.requires_grad_(False)


class TargetModel(torch.nn.Module):
    def __init__(self):
        super(TargetModel, self).__init__()
        self.diffusion = diffusion
        self.classifier = classifier
        self.eval()
        self.requires_grad_(False)
        self.device = torch.device('cuda')

    def forward(self, x):
        x = self.diffusion(x)
        x = self.classifier(x)
        return x


# attacker = DiffusionAttacker([TargetModel()], total_step=1)
attacker = DI_MI_FGSM([TargetModel()])

# for x, y in loader:
#     x, y = x[5].cuda().unsqueeze(0), y[5].cuda().unsqueeze(0)
#     original_x = x.clone()
#     # making adv_x
#     adv_x = x.clone()
#     adv_x = attacker(adv_x, y)
#
#     with torch.no_grad():
#         adv_img: Image.Image = to_img(adv_x.squeeze())
#         adv_img.save('adv.png')
#         delta = torch.abs(adv_x - original_x)
#         print(f'difference is {torch.sum(delta)}')
#         delta: Image.Image = to_img(delta.squeeze())
#         delta.save('adv-x.png')
#         # making purified image
#         purified_x = diffusion(adv_x)
#         purified_img: Image.Image = to_img(purified_x.squeeze())
#         purified_img.save('purified.png')
#         print(y)
#         # print(torch.max(classifier(adv_x), dim=1)[1])
#         print(torch.max(classifier(original_x), dim=1)[1])
#         print(torch.max(classifier(purified_x), dim=1)[1])
#         origin_img: Image.Image = to_img(original_x.squeeze())
#         origin_img.save('origin.png')
#     break

if __name__ == '__main__':
    result = test_transfer_attack_acc(attacker, loader, [TargetModel()])
   #  test_transfer_attack_acc_distributed(get_attacker, loader, get_target_model, num_gpu=4)
