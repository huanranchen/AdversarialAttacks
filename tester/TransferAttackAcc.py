import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import List, Callable, Tuple
from tqdm import tqdm
from attacks import AdversarialInputAttacker
from .utils import cosine_similarity


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


def test_transfer_attack_acc_and_cosine_similarity(attacker: AdversarialInputAttacker,
                                                   loader: DataLoader,
                                                   target_models: List[nn.Module],
                                                   device=torch.device(
                                                       'cuda' if torch.cuda.is_available() else 'cpu')
                                                   ) -> Tuple[List[float], float]:
    criterion = nn.CrossEntropyLoss()
    train_models: List[nn.Module] = attacker.models
    transfer_accs = [0] * len(target_models)
    denominator = 0
    cosine_similarities = []
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
        # calculate cosine simiarity
        train_grads, test_grads = [], []
        x.requires_grad = True
        for m in train_models:
            loss: torch.tensor = criterion(m(x.to(m.device)), y.to(m.device))
            loss.backward()
            train_grads.append(x.grad)
            x.grad = None
        for m in target_models:
            loss: torch.tensor = criterion(m(x), y)
            loss.backward()
            test_grads.append(x.grad)
            x.grad = None
        x.requires_grad = False
        train_grads, test_grads = torch.stack(train_grads), torch.stack(test_grads)
        cosine_similarities.append(cosine_similarity(train_grads, test_grads))

    transfer_accs = [i / denominator for i in transfer_accs]
    # print
    for i, model in enumerate(target_models):
        print('-' * 100)
        print(model.__class__, model.model.__class__, 1 - transfer_accs[i])
        print('-' * 100)
    cosine_similarities = sum(cosine_similarities) / len(cosine_similarities)
    print('-' * 100)
    print('the cosine similarity is ', cosine_similarities)
    print('-' * 100)
    return transfer_accs, cosine_similarities
