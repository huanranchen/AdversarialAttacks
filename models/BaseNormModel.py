import torch
from torchvision import transforms


class BaseNormModel(torch.nn.Module):
    '''
    **kwargs aims to unify with "pretrained=True"
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 transform=transforms.Compose([
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ]),
                 **kwargs):
        super(BaseNormModel, self).__init__()
        self.model = model
        self.transforms = transform
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.transforms(x)
        return self.model(x)


class Identity(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 **kwargs,
                 ):
        super(Identity, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        return self.model(x)
