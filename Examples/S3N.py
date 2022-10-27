import os
import sys
import torch
import torchvision.transforms

sys.path.append(os.path.abspath('.'))
from train import Trainer
from model.loss.S3N_loss import MultiSmoothLoss
from utils import accuracy


class S3NTrainer(Trainer):

    def __init__(self):
        super(S3NTrainer, self).__init__()

    def get_transformers(self, config):
        return {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(448, scale=(0.5, 1)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(448),
                torchvision.transforms.CenterCrop(448),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        }

    def get_criterion(self, config):
        return MultiSmoothLoss(config)

    def get_optimizer(self, config):
        model = self.get_model_module()

        classifier_params = [v if 'classifier' in k else None for k, v in model.named_parameters()]
        classifier_params = list(filter(lambda p: p is not None, classifier_params))

        # base_param_ids = list(map(id, model.features.parameters()))
        radius_param_ids = list(map(id, model.radius.parameters()))
        filter_param_ids = list(map(id, model.filter.parameters()))
        classifier_param_ids = list(map(id, classifier_params))
        ids = [*radius_param_ids, *filter_param_ids, *classifier_param_ids]

        other_params = list(filter(lambda p: id(p) not in ids, model.parameters()))

        return torch.optim.SGD([
            {'params': classifier_params, 'lr': config.lr},
            {'params': model.radius.parameters(), 'lr': 0.00001 * config.lr},
            {'params': model.filter.parameters(), 'lr': 0.00001 * config.lr},
            {'params': other_params, 'lr': 0.1 * config.lr}
        ], weight_decay=config.weight_decay)

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.T_max, config.eta_min)

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        p = 0 if self.epoch < 20 else 1
        # forward
        outputs = self.get_model_module()(images, p)
        loss = self.criterion(outputs, labels)
        aggregation, agg_origin, agg_sampler, agg_sampler1 = outputs

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(aggregation, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        p = 1 if self.epoch < 20 else 2

        aggregation, agg_origin, agg_sampler, agg_sampler1 = self.get_model_module()(images, p)
        acc = accuracy(aggregation, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = S3NTrainer()
    # print(trainer.model)
    trainer.train()
