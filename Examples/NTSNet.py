import os
import sys
import torch

sys.path.append(os.path.abspath('.'))
from model.loss.NTS_loss import NTSLoss
from train import Trainer
from utils import accuracy


class NTSTrainer(Trainer):
    def __init__(self):
        super(NTSTrainer, self).__init__()

    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def get_criterion(self, config):
        return NTSLoss(config)

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

        output = self.model(images)
        raw_logits, concat_logits, part_logits, _, top_n_prob = output
        loss = self.criterion(output, labels)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(concat_logits, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        output = self.model(images)
        raw_logits, concat_logits, part_logits, _, top_n_prob = output
        acc = accuracy(concat_logits, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = NTSTrainer()
    # print(trainer.model)
    trainer.train()
