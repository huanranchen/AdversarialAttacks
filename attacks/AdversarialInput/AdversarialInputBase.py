import torch
from abc import abstractmethod
from typing import List


class AdversarialInputAttacker():
    def __init__(self, model: List[torch.nn.Module]):
        self.models = model

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    def model_distribute(self):
        '''
        make each model on one gpu
        :return:
        '''
        assert torch.cuda.device_count() >= len(self.models), \
            'my parallel must ensure that num_gpus >= num_models'
        for i, model in enumerate(self.models):
            model.to(torch.device(f'cuda:{i}'))
