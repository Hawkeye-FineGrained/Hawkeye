import os
import sys
import torch

sys.path.append(os.path.abspath('.'))
from train import Trainer


class CBCNNTrainer(Trainer):
    def __init__(self):
        super(CBCNNTrainer, self).__init__()

        if self.config.model.stage == 1:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def get_optimizer(self, config):
        # Stage 1, freeze backbone parameters.
        if self.config.model.stage == 1:
            return torch.optim.SGD(
                self.model.classifier.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        # Stage 2, train all parameters.
        elif self.config.model.stage == 2:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )

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
    trainer = CBCNNTrainer()
    # print(trainer.model)
    trainer.train()
