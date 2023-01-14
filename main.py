from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, DiffusionAttacker, DiffusionPatchAttacker, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, SGD, DI_MI_FGSM
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt
from defenses import DiffusionPureImageNet
from torchvision import transforms
from PIL import Image
from tester import test_transfer_attack_acc, test_transfer_attack_acc_distributed

loader = get_NIPS17_loader(batch_size=128, shuffle=True, num_workers=0)
to_img = transforms.ToPILImage()
classifier = BaseNormModel(resnet50(pretrained=True)).cuda()
classifier.eval()
classifier.requires_grad_(False)
model = DiffusionPureImageNet()
diffusion = model.cuda()
diffusion.eval()
diffusion.requires_grad_(False)
attacker = DiffusionPatchAttacker([diffusion])
# attacker = DI_MI_FGSM([classifier], epsilon=4/255, total_step=300, step_size=4/255/10)

for x, y in loader:
    x, y = x[15].cuda().unsqueeze(0), y[15].cuda().unsqueeze(0)
    original_x = x.clone()
    # making adv_x
    adv_x = x.clone()
    adv_x = attacker(adv_x, y)
    adv_img: Image.Image = to_img(adv_x.squeeze())
    adv_img.save('adv.png')
    delta = torch.abs(adv_x - original_x)
    print(f'difference is {torch.sum(delta)}')
    delta: Image.Image = to_img(delta.squeeze())
    delta.save('adv-x.png')
    adv_x = Image.open('adv_x.png')
    adv_x = transforms.ToTensor()(adv_x).unsqueeze(0).cuda()

    for i in range(150, 151):
        purified_x = diffusion(adv_x, diffusion_iter_time=i)
        purified_img: Image.Image = to_img(purified_x.squeeze())
        purified_img.save(f'./what/{i}.png')

    # making purified image
    purified_x = diffusion(adv_x)
    purified_img: Image.Image = to_img(purified_x.squeeze())
    purified_img.save('purified.png')
    # origin_img: Image.Image = to_img(original_x.squeeze())
    # origin_img.save('origin.png')
    print(y, torch.max(classifier(adv_x), dim=1)[1], torch.max(classifier(purified_x), dim=1)[1])
    break


class TargetModel(torch.nn.Module):
    def __init__(self):
        super(TargetModel, self).__init__()
        self.diffusion = diffusion
        self.classifier = classifier

    def forward(self, x):
        x = self.diffusion(x)
        x = self.classifier(x)
        return x

# if __name__ == '__main__':
#     result = test_transfer_attack_acc(attacker, loader, [TargetModel()])
#    #  test_transfer_attack_acc_distributed(get_attacker, loader, get_target_model, num_gpu=4)

