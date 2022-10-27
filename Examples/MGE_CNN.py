import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath('.'))
from train import Trainer
from utils import accuracy


class MGE_CNNTrainer(Trainer):
    def __init__(self):
        super(MGE_CNNTrainer, self).__init__()

    def get_criterion(self, config):
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_optimizer(self, config):
        lr_extractor = config.lr * (0.1 if 'lr_rate' not in config else config.lr_rate)
        return torch.optim.Adam([
            {'params': self.model.get_params(prefix='classifier'), 'lr': config.lr},
            {'params': self.model.get_params(prefix='extractor'), 'lr': lr_extractor}
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
        logits = outputs['logits']

        losses = [self.criterion(logit, labels) for k, logit in enumerate(logits)]
        losses.append(sum(losses) / len(losses))
        loss = losses[-1]

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(outputs['logits'][-1], labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(losses[-1].item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        results = self.model(images)
        acc = accuracy(results['logits'][-1], labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = MGE_CNNTrainer()
    # print(trainer.model)
    trainer.train()
