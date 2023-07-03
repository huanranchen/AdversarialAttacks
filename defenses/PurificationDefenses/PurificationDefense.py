import torch
from torch import nn


class PurificationDefense(nn.Module):
    def __init__(self,
                 purifier: nn.Module,
                 classifier: nn.Module,
                 device=torch.device('cuda')):
        super(PurificationDefense, self).__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.device = device
        #
        self.init()

    def init(self):
        self.purifier.eval().requires_grad_(False).to(self.device)
        self.classifier.eval().requires_grad_(False).to(self.device)

    def forward(self, x):
        x = self.purifier(x)
        x = self.classifier(x)
        return x
