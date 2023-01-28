import torch


def cosine_similarity(x: torch.tensor, y: torch.tensor or None = None) -> float:
    if y is None:
        y = x  # self cosine similarity
    # x: N1, D     y: N2, D         where N is the number of train_models
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x, y = x / torch.norm(x, dim=1).view(-1, 1), y / torch.norm(y, dim=1).view(-1, 1)
    gram = x @ y.permute(1, 0)  # N1, N2
    return torch.mean(gram).item()


def list_mean(x: list) -> float:
    return sum(x) / len(x)
