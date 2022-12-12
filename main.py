from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness
from models import *
import torch
from torch.nn import functional as F
from tester import test_multimodel_acc_one_image, test_transfer_attack_acc
from defenses import Randomization, JPEGCompression, BitDepthReduction

loader = get_NIPS17_loader(batch_size=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

origin_train_models = [resnet18, resnet34, resnet50, resnet101]
origin_test_models = [alexnet, convnext_tiny, densenet121, efficientnet_b0, googlenet, inception_v3,
    mnasnet0_75, mobilenet_v3_small, regnet_x_400mf, shufflenet_v2_x0_5, squeezenet1_0, vgg16,
    vit_b_16, swin_s, maxvit_t, resnet152]
# defense_list = [BaseNormModel, Randomization, JPEGCompression, BitDepthReduction]
defense_list = [BaseNormModel]
train_models, test_models = [], []
for model in origin_train_models:
    model = BaseNormModel(model(pretrained=True)).to(device)
    model.eval()
    train_models.append(model)

for model in origin_test_models:
    for defender in defense_list:
        now_model = defender(model(pretrained=True)).to(device)
        now_model.eval()
        test_models.append(now_model)

attacker_list = [FGSM, BIM, MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness]
for now_attacker in attacker_list:
    attacker = now_attacker(train_models)
    print(attacker.__class__)
    test_transfer_attack_acc(attacker, loader, test_models)
