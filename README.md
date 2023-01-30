# Adversarial Attacks
Official code implement of paper **Boosting Transferability via Attacking Common Weakness**

[**Paper**](https://arxiv.org/abs/2211.09773)
| Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

---

## Install
### Environment
run 
```bash
pip install requirements.txt
```


### Data
Cifar10 will be downloaded automatically

For PACS dataset, please refer to ./data/PACS.py for install

For NIPS17 dataset, you can run \
```bash
kaggle datasets download -d google-brain/nips-2017-adversarial-learning-development-set
```
and then put it into ./resources/NIPS17

### Model Checkpoints
All the models in our paper are from torchvision, robustbench, timm library, and the checkpoints will be downloaded automatically.

We also encapsulate some models and defenses in *"./models"* and *"./defenses"*. If you want to attack them, you can download their checkpoints by yourself

---

## Code Framework

attacks: Some attack algorithms. Including VMI, VMI-CW, CW, SAM, etc.

data: loader of CIFAR, NIPS17, PACS

defenses: Some defenses algorithm

experiments: Example codes

models: Some pretrained models

optimizer: scheduler and optimizer

tester: some functions to test accuracy and attack success rate

utils: Utilities. Like draw landscape, get time, etc.

---

## Basic functions

```
tester.test_transfer_attack_acc(attacker:AdversarialInputBase, loader:DataLoader, target_models: List[nn.Module]) \
```

This function aims to get the attack success rate on loader against target models



```
attacker = xxxAttacker(train_models: List[nn.Module])
```
You can initialize attacker like this.

### example
```python
from models import resnet18, Wong2020Fast, Engstrom2019Robustness, BaseNormModel, Identity
from attacks import MI_CommonWeakness
attacker = MI_CommonWeakness([
    BaseNormModel(resnet18(pretrained=True)), # model that requires normalization
    Identity(Wong2020Fast(pretrained=True)) # model that do not need normalization
])

from tester import test_transfer_attack_acc
from data import get_NIPS17_loader
test_transfer_attack_acc(attacker, 
                         get_NIPS17_loader(), 
                         [
                             Identity(Wong2020Fast(pretrained=True)), # white box attack
                             Identity(Engstrom2019Robustness(pretrained=True)), # transfer attack
                          ]
                         )
```


### For More Example Codes
Please visit *'./experiments'* folder. There are some example codes using our framework to attack and draw landscapes. I believe you can quickly get familiar with our framework via these example codes.
