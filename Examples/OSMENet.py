import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('.'))
from dataset.sampler import BalancedBatchSampler
from model.loss.MAMC_loss import MAMCLoss
from train import Trainer
from utils import accuracy


class OSMENetTrainer(Trainer):

    def __init__(self):
        super(OSMENetTrainer, self).__init__()

    def get_dataloader(self, config):
        # OSMENet use `BalancedBatchSampler` to sample a fixed number of categories
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
        return MAMCLoss(config)

    def get_optimizer(self, config):
        model = self.get_model_module()
        backbone_param_ids = list(map(id, model.backbone.parameters()))
        fc_params = list(filter(lambda p: id(p) not in backbone_param_ids, model.parameters()))

        return torch.optim.SGD([
            {'params': model.backbone.parameters(), 'lr': 0.1 * config.lr},
            {'params': fc_params, 'lr': config.lr},
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
        # forward
        outputs = self.model(images)  # [N P C]
        pred, x_part = outputs
        loss = self.criterion(outputs, labels)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(pred, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        pred, x_part = self.model(images)
        acc = accuracy(pred, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = OSMENetTrainer()
    # print(trainer.model)
    trainer.train()
