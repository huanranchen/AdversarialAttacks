from .models.ncsnpp import NCSNpp
import yaml
import argparse

__all__ = ['get_NCSNPP']


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_diffpure_model_cifar_config():
    # config = {
    #     "sigma_min": 0.01,
    #     "sigma_max": 50,
    #     "num_scales": 1000,
    #     "beta_min": 0.1,
    #     "beta_max": 20.,
    #     "dropout": 0.1,
    #     "name": 'ncsnpp',
    #     "scale_by_sigma": False,
    #     "ema_rate": 0.9999,
    #     "normalization": 'GroupNorm',
    #     "nonlinearity": 'swish',
    #     "nf": 128,
    #     "ch_mult": [1, 2, 2, 2],  # (1, 2, 2, 2)
    #     "num_res_blocks": 8,
    #     "attn_resolutions": [16],  # (16,)
    #     "resamp_with_conv": True,
    #     "conditional": True,
    #     "fir": False,
    #     "fir_kernel": [1, 3, 3, 1],
    #     "skip_rescale": True,
    #     "resblock_type": 'biggan',
    #     "progressive": 'none',
    #     "progressive_input": 'none',
    #     "progressive_combine": 'sum',
    #     "attention_type": 'ddpm',
    #     "init_scale": 0.,
    #     "embedding_type": 'positional',
    #     "fourier_scale": 16,
    #     "conv_size": 3,
    # }
    with open('./models/unets/score_sde/cifar10.yml') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return config


def get_NCSNPP():
    config = get_diffpure_model_cifar_config()
    model = NCSNpp(config)
    return model
