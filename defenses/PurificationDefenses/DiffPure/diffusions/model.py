import torch
from ..eval_sde_adv import parse_args_and_config_imagenet, robustness_eval, parse_args_and_config_cifar
from ..guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from ..score_sde.models import utils as mutils


def get_unet(mode='cifar'):
    if mode == 'imagenet':
        args, config = parse_args_and_config_imagenet()
    elif mode == 'cifar':
        args, config = parse_args_and_config_cifar()
    if mode == 'imagenet':
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(config.model))
        print(f'model_config: {model_config}')
        model, _ = create_model_and_diffusion(**model_config)
        if not model_config['class_cond']:
            model.load_state_dict(torch.load(f'./resources/checkpoints/'
                                             f'DiffPure/256x256_diffusion_uncond.pt'))
        else:
            model.load_state_dict(torch.load('./resources/checkpoints/DiffPure/256x256_diffusion.pt'))
        if model_config['use_fp16']:
            model.convert_to_fp16()
        img_shape = (3, 256, 256)

    elif mode == 'cifar':
        img_shape = (3, 32, 32)
        print(f'model_config: {config}')
        model = mutils.create_model(config)
        model.load_state_dict(torch.load('./resources/checkpoints/DiffPure/32x32_diffusion.pth'), strict=False)

    else:
        assert False, 'We only provide CIFAR10 and ImageNet diffusion unet model'
    betas = torch.linspace(0.1 / 1000, 20 / 1000, 1000, device=torch.device('cuda'))
    return model, betas, img_shape
