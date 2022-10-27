import os
import sys
import torch
import torchvision

sys.path.append(os.path.abspath('.'))
from model.loss.InterpParts_loss import InterpPartsLoss
from train import Trainer
from utils import accuracy


class InterpPartsNetTrainer(Trainer):

    def __init__(self):
        super(InterpPartsNetTrainer, self).__init__()

    def get_transformers(self, config):
        return {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=config.resize_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.1),
                torchvision.transforms.RandomCrop(size=config.image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                torchvision.transforms.RandomErasing(config.p_erasing)
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=config.resize_size),
                torchvision.transforms.CenterCrop(size=config.image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        }

    def get_criterion(self, config):
        return self.to_device(InterpPartsLoss(config))

    def get_optimizer(self, config):
        model = self.get_model_module()

        # fix/finetune several layers
        fixed_layers = []
        finetune_layers = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        finetune_parameters = []
        scratch_parameters = []
        for name, p in model.named_parameters():

            layer_name = name.split('.')[0]
            if layer_name not in fixed_layers:
                if layer_name in finetune_layers:
                    finetune_parameters.append(p)
                else:
                    scratch_parameters.append(p)
            else:
                p.requires_grad = False

        # define the optimizer according to different param groups
        return torch.optim.SGD([
            {'params': finetune_parameters, 'lr': config.lr},
            {'params': scratch_parameters, 'lr': 20 * config.lr},
        ], weight_decay=config.weight_decay, momentum=0.9)

    def get_scheduler(self, config):
        num_iters = len(self.dataloaders['train'])
        # print(f'num_iters: {num_iters}')
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_iters * self.config.train.epoch)

    def do_scheduler_step(self):
        pass    # Do step in every batch

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        pred, _, _ = outputs

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # record accuracy and loss
        acc = accuracy(pred, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])
        pred, _, _ = self.model(images)
        acc = accuracy(pred, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = InterpPartsNetTrainer()
    # print(trainer.model)
    trainer.train()
