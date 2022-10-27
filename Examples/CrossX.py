import os
import sys
import torch
from torchvision.transforms import transforms

sys.path.append(os.path.abspath('.'))
from train import Trainer
from utils import accuracy
from model.loss.CrossX_loss import CrossXLoss


class CrossXTrainer(Trainer):
    def __init__(self):
        super(CrossXTrainer, self).__init__()

    def get_transformers(self, config):
        return {
            'train': transforms.Compose([
                transforms.Resize((600, 600)),
                transforms.RandomCrop((448, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((600, 600)),
                transforms.CenterCrop((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])}

    def get_criterion(self, config):
        return CrossXLoss(config)

    def get_optimizer(self, config):
        return torch.optim.SGD(
            self.model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.milestones, gamma=config.gamma)

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        if self.config.model.num_parts == 1:
            acc = accuracy(outputs, labels, 1)
        else:
            outputs_ulti, outputs_plty, outputs_cmbn, ulti_ftrs, plty_ftrs, cmbn_ftrs = outputs
            outs = outputs_ulti + outputs_plty + outputs_cmbn
            acc = accuracy(outs, labels, 1)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        # forward
        if self.config.model.num_parts == 1:
            outs = self.model(images)
        else:
            outputs_ulti, outputs_plty, outputs_cmbn, ulti_ftrs, plty_ftrs, cmbn_ftrs = self.model(images)
            outs = outputs_ulti + outputs_plty + outputs_cmbn

        acc = accuracy(outs, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = CrossXTrainer()
    # print(trainer.model)
    trainer.train()
