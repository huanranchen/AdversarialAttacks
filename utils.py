import torch
import torchvision.transforms
from PIL import Image
import numpy as np


@torch.no_grad()
def show_image(x: torch.tensor) -> Image.Image:
    if len(x.shape) == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0) * 255
    x = x.numpy()
    x = Image.fromarray(np.uint8(x))
    return x


@torch.no_grad()
def scale_and_show_tensor(x: torch.tensor):
    x += torch.min(x)
    x /= torch.max(x)
    return show_image(x)


def get_image(path: str = 'image.jpg') -> torch.tensor:
    image = Image.open(path)
    image = image.convert('RGB')
    transform = torchvision.transforms.ToTensor()
    return transform(image)
