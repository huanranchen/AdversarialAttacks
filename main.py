import os
from data import get_NIPS17_loader
from attacks import CosineSimilarityEncourager, SequentialAttacker, ParallelAttacker, Perturbation
from optimizer import PGD
from models import *
import torch
from tqdm import tqdm
from torch.nn import functional as F
from tester import test_multimodel_acc_one_image, test_acc

loader = get_NIPS17_loader(batch_size=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

origin_train_models = [resnet50, resnet152, resnet18, resnet101, resnet34]
origin_test_models = [wide_resnet50_2, wide_resnet101_2]

train_models, test_models = [], []
for model in origin_train_models:
    model = BaseNormModel(model(pretrained=True)).to(device)
    model.eval()
    train_models.append(model)

for model in origin_test_models:
    model = BaseNormModel(model(pretrained=True)).to(device)
    model.eval()
    test_models.append(model)

test_acc(test_models[0], loader)

# x, y = next(iter(loader))
# x, y = x.to(device), y.to(device)
# perturb = Perturbation(PGD)
# perturb.constant_init(0)
# attacker = SequentialAttacker(train_models, perturb)
# p = attacker.attack(attacker.tensor_to_loader(x, y), total_iter_step=1)
# # adv_x = x + p.perturbation
# adv_x = x
# test_multimodel_acc_one_image(x, y, test_models)
