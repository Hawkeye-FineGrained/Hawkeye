import os
import sys
import torch

sys.path.append(os.path.abspath('.'))
from train import Trainer


class MPNTrainer(Trainer):
    def __init__(self):
        super(MPNTrainer, self).__init__()

    def get_optimizer(self, config):
        return torch.optim.Adam([
            {'params': self.model.classifier.parameters(), 'lr': config.lr},
            {'params': self.model.pool.parameters(), 'lr': config.lr},
            {'params': self.model.backbone.parameters(), 'lr': 0.2 * config.lr},
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


if __name__ == '__main__':
    trainer = MPNTrainer()
    # print(trainer.model)
    trainer.train()
