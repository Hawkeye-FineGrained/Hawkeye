import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath('.'))
from model.utils import freeze, unfreeze
from train import Trainer
from utils import accuracy


class MyTrainer(Trainer):
    def __init__(self):
        super(MyTrainer, self).__init__()
        self.num_batches = len(self.dataloaders['train'])
        self.logger.info(f'num_epochs in train: {self.num_batches}')

    def get_criterion(self, config):
        return F.nll_loss

    def get_optimizer(self, config):
        model = self.get_model_module()

        # create parameter groups
        params_to_freeze = []
        params_to_train = []

        dist_params = []
        for name, param in model.named_parameters():
            if 'dist_params' in name:
                dist_params.append(param)

        # freeze resnet50 except last convolutional layer
        for name, param in model.backbone.named_parameters():
            if '7.2' not in name:                   # layer4.2
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        if config.name == 'SGD':
            param_list = [
                {"params": params_to_freeze, "lr": 0.01 * config.lr_net, "weight_decay_rate": config.weight_decay},
                {"params": params_to_train, "lr": config.lr, "weight_decay_rate": config.weight_decay},
                {"params": model.neck_conv.parameters(), "lr": config.lr, "weight_decay_rate": config.weight_decay},
                {"params": model.tree.prototype_layer.parameters(), "lr": config.lr, "weight_decay_rate": 0,
                 "momentum": 0}
            ]
            if self.config.model.disable_derivative_free_leaf_optim:
                param_list.append({"params": dist_params, "lr": config.lr_pi, "weight_decay_rate": 0})
        else:
            param_list = [
                {"params": params_to_freeze, "lr": 0.01 * config.lr, "weight_decay_rate": config.weight_decay},
                {"params": params_to_train, "lr": config.lr, "weight_decay_rate": config.weight_decay},
                {"params": model.neck_conv.parameters(), "lr": config.lr, "weight_decay_rate": config.weight_decay},
                {"params": self.model.tree.prototype_layer.parameters(), "lr": config.lr, "weight_decay_rate": 0}
            ]
            if self.config.model.disable_derivative_free_leaf_optim:
                param_list.append({"params": dist_params, "lr": config.lr_pi, "weight_decay_rate": 0})

        self.params_to_freeze = params_to_freeze
        # self.params_to_train = params_to_train

        if config.name == 'SGD':
            return torch.optim.SGD(param_list, lr=config.lr, momentum=config.momentum)
        if config.name == 'Adam':
            return torch.optim.Adam(param_list, lr=config.lr, eps=1e-07)
        if config.name == 'AdamW':
            return torch.optim.AdamW(param_list, lr=config.lr, eps=1e-07, weight_decay=config.weight_decay)

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

    def on_start_epoch(self, config):
        lrs = '  '.join([str(p['lr']) for p in self.optimizer.param_groups])
        self.logger.info(f'Epoch:{self.epoch}  lrs: {lrs}')

        # init some variables
        with torch.no_grad():
            self._old_dist_params = dict()
            for leaf in self.model.tree.leaves:
                self._old_dist_params[leaf] = leaf._dist_params.detach().clone()
            # Optimize class distributions in leafs
            self.eye = torch.eye(self.config.model.num_classes).to(self.device[0])

        # freeze net for first 30 epochs
        if self.epoch == 0:
            freeze(self.params_to_freeze)
            self.logger.info(f'Epoch:{self.epoch}  Freeze params')
        elif self.epoch == 30:
            unfreeze(self.params_to_freeze)
            self.logger.info(f'Epoch:{self.epoch}  Unfreeze params')

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])
        self.optimizer.zero_grad()

        outputs = self.model(images, labels)
        pred, info = outputs
        loss = self.criterion(torch.log(pred), labels)

        # backward
        loss.backward()
        self.optimizer.step()

        # Update leaves with derivate-free algorithm
        if not self.config.model.disable_derivative_free_leaf_optim:
            # Make sure the tree is in eval mode
            self.model.eval()
            with torch.no_grad():
                target = self.eye[labels]  # shape (batch_size, num_classes)
                for leaf in self.model.tree.leaves:
                    if self.config.model.log_probabilities:
                        # log version
                        update = torch.exp(torch.logsumexp(
                            info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - pred, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target) / pred, dim=0)
                    leaf._dist_params -= (self._old_dist_params[leaf] / self.num_batches)
                    # dist_params values can get slightly negative because of floating point issues. therefore,
                    # set to zero.
                    F.relu_(leaf._dist_params)
                    leaf._dist_params += update

        # record accuracy and loss
        acc = accuracy(pred, labels, 1)
        self.average_meters['acc'].update(acc)
        self.average_meters['loss'].update(loss.item())

        # # add histogram for test
        # for i in [2, 10, 88, 100]:
        #     self.tb_writer.add_histogram(f'leaf_node_{i}', self.model.tree.leaves[i], self.epoch)

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])
        pred, info = self.model(images)
        acc = accuracy(pred, labels, 1)
        self.average_meters['acc'].update(acc, pred.size(0))


if __name__ == '__main__':
    trainer = MyTrainer()
    # print(trainer.model)
    trainer.train()
