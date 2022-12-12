import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
from typing import List, Callable
from tqdm import tqdm


def test_transfer_attack_acc(attacker: Callable, loader: DataLoader,
                             target_models: List[nn.Module],
                             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> List[float]:
    transfer_accs = [0] * len(target_models)
    denominator = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        x = attacker(x, y)
        with torch.no_grad():
            denominator += x.shape[0]
            for i, model in enumerate(target_models):
                pre = model(x)  # N, D
                pre = torch.max(pre, dim=1)[1]  # N
                transfer_accs[i] += torch.sum(pre == y).item()

    transfer_accs = [i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print('-' * 100)
        print(model.__class__, model.model.__class__, 1 - transfer_accs[i])
        print('-' * 100)
    return transfer_accs
