import torch
from torch import nn, Tensor
from torch.autograd import Function
from defenses.PurificationDefenses.PurificationDefense import PurificationDefense
from defenses.PurificationDefenses.DiffPure.diffusions.DiffusionClassifier import DiffusionClassifier


class DiffusionLikelihoodMaximizerFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        assert x.shape[0] == 1, 'batch size should be 1'
        assert x.is_leaf, 'x must be leaf variable'
        x = DiffusionLikelihoodMaximizerFunction.classifier.optimize_back(x,
                                                                          y=torch.tensor([0]).cuda(),
                                                                          )
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


def diffusion_likelihood_maximizer_defense(classifier: nn.Module):
    DiffusionLikelihoodMaximizerFunction.classifier = DiffusionClassifier()

    class Purifier(nn.Module):
        def __init__(self):
            super(Purifier, self).__init__()
            self.classifier = DiffusionLikelihoodMaximizerFunction.classifier
            self.classifier.unet.load_state_dict(torch.load('./unet_condition_old.pt'))

        def forward(self, x):
            return DiffusionLikelihoodMaximizerFunction.apply(x)

    return PurificationDefense(Purifier(), classifier)
