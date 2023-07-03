import torch
from torchvision import transforms
import numpy as np
import random
from defenses.PurificationDefenses.DiffPure import DiffusionClassifier
from defenses.PurificationDefenses.DiffPure.guided_diffusion.unet import UNetModel
from utils.ImageHandling import save_image

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
device = torch.device('cuda')

unet = UNetModel(image_size=32, in_channels=1, out_channels=1,
                 model_channels=64, num_res_blocks=2, channel_mult=(1, 2, 3, 4),
                 attention_resolutions=[8, 4], num_heads=4,
                 num_classes=10)
unet.load_state_dict(torch.load('unet_mnist.pt'))

diffpure = DiffusionClassifier(unet=unet)
diffpure.eval().requires_grad_(False).to(device)
total_images = 1
for i in range(total_images):
    x = diffpure.generation(total_images=1, class_id=random.randint(0, 9), iter_each_sample=200)
    now_x = x[0]
    save_image(now_x, path=f'./mcmc/{i}.png')
