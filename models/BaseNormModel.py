import torch
from torchvision import transforms


class BaseNormModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, transform=transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])):
        self.model = model
        self.transforms = transform
        super(BaseNormModel, self).__init__()

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)
