from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, \
    DI_MI_FGSM, VMI_FGSM, MI_TI_FGSM
from models import *
import torch
from torch import nn
from torch.nn import functional as F
from tester import test_multimodel_acc_one_image, test_transfer_attack_acc, \
    test_transfer_attack_acc_and_cosine_similarity
from defenses import Randomization, JPEGCompression, BitDepthReduction, \
    NeuralRepresentationPurifier, randomized_smoothing_resnet50

loader = get_NIPS17_loader(batch_size=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

origin_train_models = [BaseNormModel(resnet18(pretrained=True)),
                       BaseNormModel(resnet34(pretrained=True)),
                       BaseNormModel(resnet50(pretrained=True)),
                       BaseNormModel(resnet101(pretrained=True)),
                       Identity(Salman2020Do_R50()),
                       Identity(Debenedetti2022Light_XCiT_S12()),
                       ]
train_models, test_models = [], []

for model in origin_train_models:
    model.eval().to(device)
    model.requires_grad_(False)
    train_models.append(model)

origin_test_models = [alexnet, convnext_tiny, densenet121, efficientnet_b0, googlenet, inception_v3,
                      mnasnet0_75, mobilenet_v3_small, regnet_x_400mf, shufflenet_v2_x0_5, squeezenet1_0, vgg16,
                      vit_b_16, swin_s, maxvit_t, resnet152, adv_inception_v3, ens_adv_inception_resnet_v2]
for model in origin_test_models:
    now_model = BaseNormModel(model(pretrained=True)).to(device)
    now_model.eval()
    now_model.requires_grad_(False)
    test_models.append(now_model)

origin_test_models = [Wong2020Fast, Engstrom2019Robustness,
                      Salman2020Do_R18, Salman2020Do_50_2,
                      Debenedetti2022Light_XCiT_M12, Debenedetti2022Light_XCiT_L12]
for model in origin_test_models:
    now_model = Identity(model(pretrained=True)).to(device)
    now_model.eval()
    now_model.requires_grad_(False)
    test_models.append(now_model)

attacker_list = [MI_SAM, MI_CosineSimilarityEncourager, MI_CommonWeakness]
for now_attacker in attacker_list:
    attacker = now_attacker(train_models)
    print(attacker.__class__)
    test_transfer_attack_acc(attacker, loader, test_models)
