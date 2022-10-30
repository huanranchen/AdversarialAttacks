from torch.utils.data import Dataset, DataLoader


def get_loader(dataset: Dataset,
               batch_size=32,
               shuffle=True,
               ):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
