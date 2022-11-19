import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Iterable


@torch.no_grad()
def test_acc(model: nn.Module, loader: DataLoader or Iterable,
             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    total_loss = 0
    total_acc = 0
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device).eval()
    denominator = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pre = model(x)
        total_loss += criterion(pre, y).item() * y.shape[0]
        _, pre = torch.max(pre, dim=1)
        total_acc += torch.sum((pre == y)).item()
        denominator += y.shape[0]

    test_loss = total_loss / denominator
    test_accuracy = total_acc / denominator
    print(f'loss = {test_loss}, acc = {test_accuracy}')
    return test_loss, test_accuracy
