from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, SGD
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt

loader = get_NIPS17_loader(batch_size=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# origin_train_models = [resnet152]
# origin_test_models = [resnet18]
train_models = [BaseNormModel(resnet18(pretrained=True)).eval(),
                BaseNormModel(resnet34(pretrained=True)).eval(),
                BaseNormModel(resnet50(pretrained=True)).eval(),
                BaseNormModel(resnet101(pretrained=True)).eval(),
                Salman2020Do_R50().eval(),
                Debenedetti2022Light_XCiT_S12().eval()
                ]
for m in train_models:
    m.eval()
    m.requires_grad_(False)
origin_test_models = [Wong2020Fast, Engstrom2019Robustness,
                      Salman2020Do_R18, Salman2020Do_50_2,
                      Debenedetti2022Light_XCiT_M12, Debenedetti2022Light_XCiT_L12]

test_models = []

for model in origin_test_models:
    model = Identity(model(pretrained=True)).to(device)
    model.eval()
    model.requires_grad_(False)
    test_models.append(model)
# test_models = train_models

# test_models += train_models

attacker = MI_FGSM(train_models, targeted_attack=True)
x, y = next(iter(loader))
x, y = x.cuda(), y.cuda()
original_x = x.clone()
x = attacker(x, y)
test_models = [m.to(torch.device('cuda')) for m in test_models]

for i in range(len(test_models)):
    finetuner = BIM([test_models[i]], step_size=16/255/10, targeted_attack=True)
    now_x = finetuner(x, y)
    drawer = Landscape4Input(lambda x: F.cross_entropy(test_models[i](x), y.cuda()).item(),
                             input=x.cuda(), mode='2D')
    drawer.synthesize_coordinates(x_min=-16 / 255, x_max=16 / 255, x_interval=1 / 255)
    direction = (now_x - x)
    direction /= torch.norm(direction, p=float('inf'))
    drawer.assign_unit_vector(direction)
    drawer.draw()
legends = ['Wong2020Fast', 'Engstrom2019Robustness', 'Salman2020Do_R18',
           'Salman2020Do_50_2', 'Debenedetti2022Light_XCiT_M12', 'Debenedetti2022Light_XCiT_L12']
plt.legend(legends)

plt.show()
