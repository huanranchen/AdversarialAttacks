from data import get_NIPS17_loader
from attacks import BIM, FGSM, PGD, MI_RandomWeight, \
    MI_FGSM, MI_CosineSimilarityEncourager, MI_SAM, MI_CommonWeakness, SGD
from models import *
import torch
from utils import Landscape4Input
from torch.nn import functional as F
from matplotlib import pyplot as plt
from defenses import get_diffpure_imagenet

model = get_diffpure_imagenet()
diffusion = model.runner