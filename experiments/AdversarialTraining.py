from models import WideResNet_70_16_dropout
from data import get_CIFAR10_train, get_CIFAR10_test
from defenses import AdversarialTraining
from attacks import FGSM, BIM, PGD

train_loader = get_CIFAR10_train(batch_size=128, augment=False)
eval_loader = get_CIFAR10_test(batch_size=32)
model = WideResNet_70_16_dropout()
fgsm_attacker = FGSM([model], epsilon=8 / 255, step_size=8 / 255, random_start=True)
pgd_attacker = PGD([model], epsilon=8 / 255, step_size=8 / 255 / 10)
solver = AdversarialTraining(pgd_attacker, model, writer_name='overfittingplay_no_aug')
solver.train(train_loader, eval_loader=eval_loader, test_attacker=pgd_attacker)
