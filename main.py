import os
from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, MI_FGSM, MI_CosineSimilarityEncourager
from models import *
import torch
from torch.nn import functional as F
from tester import test_multimodel_acc_one_image, test_transfer_attack_acc

loader = get_NIPS17_loader(batch_size=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

origin_train_models = [vgg16, inception_v3, convnext_tiny, regnet_x_400mf,
                       resnet152, shufflenet_v2_x0_5, mnasnet0_5, efficientnet_b0]
origin_test_models = [swin_s]
# origin_train_models = [inception_v4, inception_resnet_v2, resnet152]
# origin_test_models = [ inception_v3, ]

train_models, test_models = [], []
for model in origin_train_models:
    model = BaseNormModel(model(pretrained=True)).to(device)
    model.eval()
    train_models.append(model)

for model in origin_test_models:
    model = BaseNormModel(model(pretrained=True)).to(device)
    model.eval()
    test_models.append(model)

# x, y = next(iter(loader))
# x, y = x.to(device), y.to(device)
# perturb = Perturbation(PGD)
# perturb.constant_init(0)
# attacker = CosineSimilarityEncourager(train_models, perturb, outer_optimizer=lambda x: PGD(x, lr=0.9))
# p = attacker.attack(attacker.tensor_to_loader(x, y), total_iter_step=10)
# adv_x = x + p.perturbation
# test_multimodel_acc_one_image(x, y, test_models)

attacker = MI_RandomWeight(train_models)
test_transfer_attack_acc(attacker, loader, test_models)
