import torch
import torchvision.transforms
from PIL import Image
import numpy as np



def show_image(x: torch.tensor):
    if len(x.shape) == 4: x = x.squeeze(0)
    x = x.permute(1, 2, 0) * 255
    x = x.numpy()
    x = Image.fromarray(np.uint8(x))
    x.show()


def get_image(path: str = 'image.jpg') -> torch.tensor:
    image = Image.open(path)
    image = image.convert('RGB')
    transform = torchvision.transforms.ToTensor()
    return transform(image)