import torch
import torchvision.transforms
from PIL import Image
import numpy as np
import os
from torch import Tensor
import cv2


@torch.no_grad()
def show_image(x: Tensor) -> Image.Image:
    if len(x.shape) == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0) * 255
    x = x.cpu().numpy()
    x = Image.fromarray(np.uint8(x))
    return x


@torch.no_grad()
def save_image(x: Tensor, path='./0.png') -> Image.Image:
    if len(x.shape) == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0) * 255
    x = x.cpu().numpy()
    if x.shape[2] == 1:
        cv2.imwrite(path, x.squeeze())
        return x
    x = Image.fromarray(np.uint8(x))
    x.save(path)
    return x


@torch.no_grad()
def scale_and_show_tensor(x: Tensor):
    x = x.cpu()
    x += torch.min(x)
    x /= torch.max(x)
    return show_image(x)


def get_image(path: str = 'image.jpg') -> Tensor:
    image = Image.open(path)
    image = image.convert('RGB')
    transform = torchvision.transforms.ToTensor()
    return transform(image)


def concatenate_image(img_path: str = './generated',
                      padding=1,
                      img_shape=(32, 32, 3),
                      row=10,
                      col=10,
                      save_path='concated.png',
                      ):
    imgs = os.listdir(img_path)
    assert len(imgs) >= row * col, 'images should be enough for demonstration'
    alls = []
    for img in imgs:
        img = Image.open(os.path.join(img_path, img))
        x = np.array(img)
        x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)))
        alls.append(x)
    alls = alls[:row * col]
    x = np.stack(alls)
    x = x.reshape((row, col, img_shape[0] + padding * 2, img_shape[1] + padding * 2, img_shape[2]))
    x = torch.from_numpy(x)
    x = x.permute(0, 2, 1, 3, 4).reshape(
        row * (img_shape[0] + padding * 2), col * (img_shape[1] + padding * 2),
        img_shape[2]).numpy()
    x = Image.fromarray(x)
    x.save(save_path)


def total_variation(x):
    adv_patch = x
    if len(x.shape) == 3:
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
    elif len(x.shape) == 4:
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, :, 1:] - adv_patch[:, :, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(tvcomp1)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, :, 1:, :] - adv_patch[:, :, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(tvcomp2)
        tv = tvcomp1 + tvcomp2
    else:
        raise ValueError
    return tv / torch.numel(adv_patch)
