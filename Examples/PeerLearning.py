import os
import sys
import torch
import numpy as np
from PIL import ImageFile

sys.path.append(os.path.abspath('.'))
from train import Trainer
from model.loss.peer_learning_loss import PeerLearningLoss
from utils import accuracy, AverageMeter, PerformanceMeter

# Solution of `IOError: image file is truncated`
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PLTrainer(Trainer):
    def __init__(self):
        super(PLTrainer, self).__init__()
        # The auxiliary scheduler for drop rate.
        drop_rate = self.config.model.drop_rate  # The maximum drop rate, `\xi` in Eqn.(2).
        T_k = self.config.model.T_k  # The number of epochs after which d(T) is no longer updated, `T_k` in Eqn.(2).
        self.rate_scheduler = np.ones(self.config.train.epoch) * drop_rate
        self.rate_scheduler[:T_k] = np.linspace(0, drop_rate, T_k)

    def get_performance_meters(self):
        return {
            'train': {
                metric: PerformanceMeter(higher_is_better=False if metric.startswith('loss') else True)
                for metric in ['acc', 'acc1', 'acc2', 'loss1', 'loss2']
            },
            'val': {metric: PerformanceMeter() for metric in ['acc', 'acc1', 'acc2']},
            'val_first': {metric: PerformanceMeter() for metric in ['acc']}
        }

    def get_average_meters(self):
        meters = ['acc', 'acc1', 'acc2', 'loss1', 'loss2']
        return {
            meter: AverageMeter() for meter in meters
        }

    def get_optimizer(self, config):
        stage = self.config.model.stage if 'stage' in self.config.model else None
        model = self.get_model_module()
        if stage == 1:
            optimizer = torch.optim.Adam([
                {'params': model.classifier.parameters()},
            ], lr=config.lr, weight_decay=config.weight_decay)
        elif stage is None or stage == 2:
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
            ], lr=config.lr, weight_decay=config.weight_decay)
        else:
            raise NotImplementedError()
        return optimizer

    def get_scheduler(self, config):
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['T_max'] - config['warmup_epochs']
        )
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=config['lr_warmup_decay'], total_iters=config['warmup_epochs']
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[config['warmup_epochs']]
        )
        return scheduler

    def get_criterion(self, config):
        return PeerLearningLoss

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        logits1, logits2 = self.model(images)

        loss1, loss2 = self.criterion(logits1, logits2, labels, drop_rate=self.rate_scheduler[self.epoch])

        # backward
        self.optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc1 = accuracy(logits1, labels, 1)
        acc2 = accuracy(logits2, labels, 1)
        self.average_meters['acc'].update(max(acc1, acc2), images.size(0))
        self.average_meters['acc1'].update(acc1, images.size(0))
        self.average_meters['acc2'].update(acc2, images.size(0))
        self.average_meters['loss1'].update(loss1.item(), images.size(0))
        self.average_meters['loss2'].update(loss2.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])
        logits1, logits2 = self.model(images)

        acc1 = accuracy(logits1, labels, 1)
        acc2 = accuracy(logits2, labels, 1)
        self.average_meters['acc'].update(max(acc1, acc2), images.size(0))
        self.average_meters['acc1'].update(acc1, images.size(0))
        self.average_meters['acc2'].update(acc2, images.size(0))

    def update_performance_meter(self, split):
        if split == 'train':
            self.performance_meters['train']['acc'].update(self.average_meters['acc'].avg)
            self.performance_meters['train']['acc1'].update(self.average_meters['acc1'].avg)
            self.performance_meters['train']['acc2'].update(self.average_meters['acc2'].avg)
            self.performance_meters['train']['loss1'].update(self.average_meters['loss1'].avg)
            self.performance_meters['train']['loss2'].update(self.average_meters['loss2'].avg)
        elif split == 'val':
            self.performance_meters['val']['acc'].update(self.average_meters['acc'].avg)
            self.performance_meters['val']['acc1'].update(self.average_meters['acc1'].avg)
            self.performance_meters['val']['acc2'].update(self.average_meters['acc2'].avg)


if __name__ == '__main__':
    trainer = PLTrainer()
    # print(trainer.model)
    trainer.train()
