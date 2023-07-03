from robustbench.utils import load_model
from robustbench.model_zoo.cifar10 import cifar_10_models


def Andriushchenko2020Understanding(pretrained=True):
    return load_model(model_name="Andriushchenko2020Understanding", dataset="cifar10", threat_model="Linf")


def Carmon2019Unlabeled(pretrained=True):
    return load_model(model_name="Carmon2019Unlabeled", dataset="cifar10", threat_model="Linf")


def Sehwag2020Hydra(pretrained=True):
    return load_model(model_name="Sehwag2020Hydra", dataset="cifar10", threat_model="Linf")


def Wang2020Improving(pretrained=True):
    return load_model(model_name="Wang2020Improving", dataset="cifar10", threat_model="Linf")


def Hendrycks2019Using(pretrained=True):
    return load_model(model_name="Hendrycks2019Using", dataset="cifar10", threat_model="Linf")


################## Below Are L2 Models ##########################################

def Rice2020OverfittingNetL2(pretrained=True):
    return load_model(model_name="Rice2020Overfitting", dataset="cifar10", threat_model="L2")
