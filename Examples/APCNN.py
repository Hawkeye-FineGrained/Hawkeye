import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

sys.path.append(os.path.abspath('.'))
from train import Trainer
from utils import accuracy


class APCNNTrainer(Trainer):
    def __init__(self):
        super(APCNNTrainer, self).__init__()

    def get_transformers(self, config):
        return {
            'train': transforms.Compose([
                transforms.Resize((config.resize_size, config.resize_size)),
                transforms.RandomCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                autoaugment.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize((config.resize_size, config.resize_size)),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        }

    def get_optimizer(self, config):
        model = self.get_model_module()
        return torch.optim.SGD(
            [
                {'params': nn.Sequential(*list(model.children())[7:]).parameters(), 'lr': config.lr},
                {'params': nn.Sequential(*list(model.children())[:7]).parameters(), 'lr': config.lr / 10}
            ], momentum=0.9, weight_decay=config.weight_decay)

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        outputs = self.model(images, labels)
        logits_mean, logits_list, mask_cat, roi_list = outputs
        loss = sum([self.criterion(logit, labels) for logit in logits_list])

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(logits_mean, labels, 1)
        self.average_meters['acc'].update(acc)
        self.average_meters['loss'].update(loss.item())

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])
        outputs = self.model(images, labels)
        logits_mean, logits_list, mask_cat, roi_list = outputs

        acc = accuracy(logits_mean, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))

    def on_start_epoch(self, config):
        num_epoch = self.config.train.epoch
        learning_rate = self.config.train.optimizer.lr

        def cosine_anneal_schedule(t):
            cos_inner = np.pi * (t % num_epoch)
            cos_inner /= num_epoch
            cos_out = np.cos(cos_inner) + 1
            return float(learning_rate / 2 * cos_out)

        # take over do_scheduler_step
        self.optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(self.epoch)
        self.optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(self.epoch) / 10

        lrs_str = "  ".join([f'{p["lr"]}' for p in self.optimizer.param_groups])
        self.logger.info(f'Epoch:{self.epoch}  lrs: {lrs_str}')

    def get_scheduler(self, config):
        return None

    def do_scheduler_step(self):
        pass


if __name__ == '__main__':
    trainer = APCNNTrainer()
    # print(trainer.model)
    trainer.train()
