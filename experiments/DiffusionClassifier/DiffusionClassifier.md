# Diffusion Classifier

---

## Install


### Model Checkpoints
please follow our baseline DiffPure(https://diffpure.github.io/) to download checkpoint of diffusion model.

Besides, you need to provide a conditional diffusion model on cifar10 dataset with NCSN++ achitecture, and rename to "unet_condition_old.pt"
Or train the conditional diffusion by run './experiments/TrainDiffusion.py'



---

## Usage

### Code Framework

> attacks: Some attack algorithms. Including VMI, VMI-CW, CW, SAM, etc.
> data: loader of CIFAR, NIPS17, PACS    
> defenses: Some defenses algorithm    
> experiments: Example codes    
> models: Some pretrained models   
> optimizer: scheduler and optimizer   
> tester: some functions to test accuracy and attack success rate   
> utils: Utilities. Like draw landscape, get time, HRNet, etc.     


### Basic functions

```
tester.test_transfer_attack_acc(attacker:AdversarialInputBase, loader:DataLoader, target_models: List[nn.Module]) \
```

This function aims to get the attack success rate on loader against target models



```
attacker = xxxAttacker(train_models: List[nn.Module])
```
You can initialize attacker like this.

### Examples
For more example codes, please visit *'./experiments'* folder. There are some example codes using our framework to attack and draw landscapes. I believe you can quickly get familiar with our framework via these example codes.



### Experiments
All experiments codes are in *'./experiments/DiffusionClassifier'*

