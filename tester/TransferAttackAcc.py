import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import List, Callable
from tqdm import tqdm


def test_transfer_attack_acc(attacker: Callable, loader: DataLoader,
                             target_models: List[nn.Module],
                             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> List[float]:
    transfer_accs = [0] * len(target_models)
    denominator = 0
    # count = 0
    # to_img = transforms.ToPILImage()
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        ori_x = x.clone()
        x = attacker(x, y)
        # temp = to_img(x[0])
        # temp.save(f'./what/adv_{count}.png')
        # temp = to_img(ori_x[0])
        # temp.save(f'./what/ori_{count}.png')
        # temp = to_img(x[0]-ori_x[0])
        # temp.save(f'./what/perturb_{count}.png')
        # count += 1
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
