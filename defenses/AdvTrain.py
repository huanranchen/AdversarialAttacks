from attacks import AdversarialInputAttacker
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable
from optimizer import default_optimizer


class AdversarialTraining():
    def __init__(self, attacker: AdversarialInputAttacker, model: nn.Module,
                 criterion=nn.CrossEntropyLoss(),
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 optimizer: Callable = default_optimizer):
        self.attacker = attacker
        self.student = model
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer(self.student, lr=1e-3)

    def train(self,
              loader: DataLoader,
              total_epoch=90,
              fp16=False,
              ):
        '''

        :param total_epoch:
        :param step_each_epoch: this 2 parameters is just a convention, for when output loss and acc, etc.
        :param fp16:
        :param generating_data_configuration:
        :return:
        '''
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        self.student.train()
        for epoch in range(1, total_epoch + 1):
            train_loss = 0
            train_acc = 0
            pbar = tqdm(loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)

                if fp16:
                    with autocast():
                        student_out = self.student(x)  # N, 60
                        _, pre = torch.max(student_out, dim=1)
                        loss = self.criterion(student_out, y)
                else:
                    student_out = self.student(x)  # N, 60
                    _, pre = torch.max(student_out, dim=1)
                    loss = self.criterion(student_out, y)

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()
                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    self.optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={train_loss / step}, acc={train_acc / step}')

            train_loss /= len(loader)
            train_acc /= len(loader)

            # self.scheduler.step(train_loss)

            print(f'epoch {epoch}, loss = {train_loss}, acc = {train_acc}')
            torch.save(self.student.state_dict(), 'student.pth')
