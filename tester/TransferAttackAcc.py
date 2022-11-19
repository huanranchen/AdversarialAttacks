import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
from typing import List

sys.path.append('./')
from attacks import BaseAttacker


def test_transfer_attack_acc(attacker: BaseAttacker, loader: DataLoader,
                             target_models: List[nn.Module]) -> List[float]:
    transfer_accs = [0] * len(target_models)
    denominator = 0
    for x, y in loader:
        x = attacker(x, y)
        with torch.no_grad():
            for i, model in enumerate(target_models):
                x = model(x)  # N, D
                denominator += x.shape[0]
                x = torch.max(x, dim=1)[1]  # N
                transfer_accs[i] += x == y

    transfer_accs = [i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print('-' * 100)
        print(model.__class__, transfer_accs[i])
        print('-' * 100)
    return transfer_accs
