import os
import sys
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.abspath('.'))
from dataset.sampler import BalancedBatchSampler
from model.loss.APINet_loss import APINetLoss
from train import Trainer
from utils import accuracy


class APINetTrainer(Trainer):
    def __init__(self):
        super(APINetTrainer, self).__init__()

    def get_dataloader(self, config):
        # APINet use `BalancedBatchSampler` to sample a fixed number of categories
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
        return APINetLoss(config)

    def get_optimizer(self, config):
        model = self.get_model_module()
        backbone_param_ids = list(map(id, model.backbone.parameters()))
        fc_params = list(filter(lambda p: id(p) not in backbone_param_ids, model.parameters()))
        return torch.optim.Adam(
            [
                {'params': model.backbone.parameters(), 'lr': config.lr},
                {'params': fc_params, 'lr': config.lr}
            ], weight_decay=config.weight_decay
        )

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

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        outputs = self.model(images, labels, flag='train')
        # logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2 = outputs
        self_logits, other_logits, labels1, labels2 = outputs
        loss = self.criterion(outputs, labels)
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(logits, targets, 1)
        batch_size = self_logits.shape[0] // 2
        self.average_meters['acc'].update(acc, 4 * batch_size)
        self.average_meters['loss'].update(loss.item(), 2 * batch_size)

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        logits = self.model(images, flag='val')
        acc = accuracy(logits, labels, 1)
        self.average_meters['acc'].update(acc, logits.size(0))

    def on_start_epoch(self, config):
        if self.epoch == 0:
            self.optimizer.param_groups[0]['lr'] = 0
            self.logger.info('Freeze conv')
        elif self.epoch == 8:
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']
            self.logger.info('Unfreeze conv')

        Trainer.on_start_epoch(self, config)


if __name__ == '__main__':
    trainer = APINetTrainer()
    # print(trainer.model)
    trainer.train()
