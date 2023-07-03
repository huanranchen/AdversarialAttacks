import torch


def cosine_similarity(x: list):
    '''
    input a list of tensor with same shape. return the mean cosine_similarity
    '''
    x = torch.stack(x)
    N = x.shape[0]
    x = x.reshape(N, -1)

    norm = torch.norm(x, p=2, dim=1)
    x /= norm.reshape(-1, 1)  # N, D
    similarity = x @ x.T  # N, N
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=0).to(torch.bool)
    similarity = similarity[mask]
    return torch.mean(similarity).item()


def list_mean(x: list) -> float:
    return sum(x) / len(x)
