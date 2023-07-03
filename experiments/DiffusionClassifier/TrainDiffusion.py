from models.unets import get_NCSNPP
from data import get_CIFAR10_train
from defenses.PurificationDefenses.DiffPure.solver import ConditionSolver

model = get_NCSNPP().cuda()
loader = get_CIFAR10_train(augment=True, batch_size=256)
solver = ConditionSolver(model)
solver.train(loader, total_epoch=2000, p_uncondition=0.1)
