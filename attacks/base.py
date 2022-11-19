import torch
from abc import abstractmethod


class BaseAttacker():
    def __init__(self):
        pass

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)
