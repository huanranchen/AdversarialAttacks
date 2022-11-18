from torch.utils.data import Dataset, DataLoader


__kaggle_link__ = 'kaggle datasets download -d google-brain/nips-2017-adversarial-learning-development-set'


class NIPS17(Dataset):
    def __init__(self, images_path = './resources/NIPS17/images'):
