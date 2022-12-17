from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F

loader = get_NIPS17_loader(batch_size=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# origin_train_models = [resnet152]
# origin_test_models = [resnet18]
origin_train_models = [resnet18, resnet34, resnet50, resnet101]
origin_test_models = [swin_s]

train_models, test_models = [], []
for model in origin_train_models:
    model = BaseNormModel(model(pretrained=True)).to(device)
    model.eval()
    train_models.append(model)

for model in origin_test_models:
    model = BaseNormModel(model(pretrained=True)).to(device)
    model.eval()
    test_models.append(model)


attacker = MI_SAM(train_models)
x, y = next(iter(loader))
x = attacker(x, y)
drawer = Landscape4Input(lambda x: F.cross_entropy(test_models[0](x), y.cuda()).item(),
                         input=x.cuda(), mode='2D')
drawer.synthesize_coordinates()
drawer.draw()