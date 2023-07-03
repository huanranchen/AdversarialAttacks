import torch
from attacks import BIM
from data import get_CIFAR10_test
from tester import test_transfer_attack_acc
from defenses import DiffusionPure
import numpy as np
import random

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

loader = get_CIFAR10_test(batch_size=1)
device = torch.device('cuda')

diffpure = DiffusionPure(mode='sde', grad_checkpoint=True).eval().requires_grad_(False)

loader = [(x, y) for i, ((x, y)) in enumerate(loader) if 0 <= i < 256]
attacker = BIM([diffpure], step_size=1 / 255, total_step=80, eot_step=1024, epsilon=8 / 255,
               norm='Linf',
               eot_batch_size=64)
test_transfer_attack_acc(
    attacker,
    loader[:64],
    [diffpure],
)
