import torch
from torch.utils.data import DataLoader
from typing import Iterable, Dict, Tuple
from tqdm import tqdm
from defenses.RandomizedSmoothing import Smooth

__WRONG_PREDICTION__ = -1


@torch.no_grad()
def certify_robustness(model: Smooth, loader: DataLoader or Iterable,
                       epsilons: Iterable = (0, 0.25, 0.5, 0.75, 1),
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                       ) -> Tuple[Iterable, Dict]:
    model.base_classifier.to(device).eval()
    radii = []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        label, radius = model.certify(x.squeeze())
        radii.append(radius if label == y.item() else __WRONG_PREDICTION__)
    radii_tensor = torch.tensor(radii)
    denominator = len(radii)
    result = dict()
    for eps in epsilons:
        print('-' * 100)
        result[eps] = torch.sum(radii_tensor >= eps).item() / denominator
        print(f'certified robustness at {eps} is {result[eps]}')
        print('-' * 100)
    return radii, result
