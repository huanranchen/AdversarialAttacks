import torch
from data import get_CIFAR10_test
from torchvision import transforms
import numpy as np
import random


torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

to_img = transforms.ToPILImage()
loader = get_CIFAR10_test(batch_size=1)
device = torch.device('cuda')

beta = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=device)
alpha = (1 - beta)
alpha_bar = alpha.cumprod(dim=0)


def save_img(x, name='test'):
    if x.ndim == 4:
        x = x[0]
    img = to_img(x)
    img.save(f'{name}.png')


# diffpure = KnnDiffusionClassifier()
# diffpure.eval().requires_grad_(False).to(device)
# x = diffpure.generation(total_images=1)
# for i, now_x in enumerate(x):
#     save_img(now_x, name=f'./mcmc/{i}.png')

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DModel
import torch

repo_id = "stabilityai/stable-diffusion-2-base"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float32, revision="fp16")
pipe.to(device)
unet = pipe.unet
decoder = pipe.vae.decoder
scheduler = pipe.scheduler
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

from tqdm import tqdm

x = torch.randn(1, 4, 64, 64, device=device)
x.requires_grad = False
optimizer = torch.optim.Adam([x], lr=1e-1)
prompt = ["A image of a pineapple"]
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)
text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * 1, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

with torch.no_grad():
    for _ in tqdm(range(1000)):
        t = torch.randint(low=20, high=980, size=(1,), device=device)
        noise = torch.randn_like(x)
        # noised_x = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1) * x + \
        #            torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1) * noise
        noised_x = scheduler.add_noise(x, noise, t)
        latent_model_input = torch.cat([noised_x] * 2, dim=0)
        pre = unet(latent_model_input, t, text_embeddings).sample
        noise_pred_uncond, noise_pred_text = pre.chunk(2)
        pre = noise_pred_uncond + 100 * (noise_pred_text - noise_pred_uncond)
        optimizer.zero_grad()
        x.grad = pre - noise
        ori_x = x.clone()
        optimizer.step()
        print(torch.norm(x - ori_x), optimizer.param_groups[0]['lr'])
        optimizer.param_groups[0]['lr'] *= 0.99
        x = x.clamp_(-1, 1)
    # scale and decode the image latents with vae
    x = 1 / 0.18215 * x
    x = decoder(x)
    print(x)
    print(x.shape)
    x = x[:, :3, :, :]
    x = (x / 2 + 0.5).clamp(0, 1)
    save_img(x)
