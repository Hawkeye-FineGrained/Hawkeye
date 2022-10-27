import os
import sys
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.abspath('.'))
from dataset.sampler import BalancedBatchSampler
from model.loss.CIN_loss import CINLoss
from train import Trainer
from utils import accuracy


class CINTrainer(Trainer):
    def __init__(self):
        super(CINTrainer, self).__init__()

    def get_dataloader(self, config):
        # CIN use `BalancedBatchSampler` to sample a fixed number of categories
        # and a fixed number of samples in each category.
        train_sampler = BalancedBatchSampler(self.datasets['train'], config.n_classes, config.n_samples)
        dataloaders = {
            'train': DataLoader(
                self.datasets['train'], num_workers=config.num_workers, pin_memory=True, batch_sampler=train_sampler
            ),
            'val': DataLoader(
                self.datasets['val'], batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True
            )
        }
        return dataloaders

    def get_criterion(self, config):
        return self.to_device(CINLoss(config))

    def get_optimizer(self, config):
        return torch.optim.SGD(
            [
                {'params': self.model.parameters(), 'lr': config.lr},
                # CIN use a linear layer when it computes loss.
                {'params': self.criterion.parameters(), 'lr': config.lr},
            ], weight_decay=config.weight_decay)

    def get_scheduler(self, config):
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['T_max'] - config['warmup_epochs']
        )
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=config['lr_warmup_decay'], total_iters=config['warmup_epochs']
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config['warmup_epochs']]
        )
        return scheduler

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        logits, _ = outputs

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(logits, labels, 1)
        self.average_meters['acc'].update(acc)
        self.average_meters['loss'].update(loss.item())


if __name__ == '__main__':
    trainer = CINTrainer()
    # print(trainer.model)
    trainer.train()
