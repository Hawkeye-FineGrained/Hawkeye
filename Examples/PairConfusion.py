import os
import sys
import torch

sys.path.append(os.path.abspath('.'))
from model.loss.pair_confusion import PairwiseConfusionLoss
from train import Trainer


class PCResNetTrainer(Trainer):
    def __init__(self):
        super(PCResNetTrainer, self).__init__()

    def get_criterion(self, config):
        return PairwiseConfusionLoss(config)

    def get_optimizer(self, config):
        classifier_param_ids = list(map(id, self.model.fc.parameters()))
        base_params = list(filter(lambda p: id(p) not in classifier_param_ids, self.model.parameters()))
        return torch.optim.Adam([
            {'params': self.model.fc.parameters(), 'lr': config.lr},
            {'params': base_params, 'lr': 0.1 * config.lr}
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
    trainer = PCResNetTrainer()
    # print(trainer.model)
    trainer.train()
