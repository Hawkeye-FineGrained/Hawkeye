import torch
import torch.nn as nn
from torchvision import transforms
from train import Trainer
from utils import accuracy


class BaselineTrainer(Trainer):
    def __init__(self):
        super(BaselineTrainer, self).__init__()

    def get_transformers(self, config):
        transformers = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        }
        return transformers

    def get_criterion(self, config):
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), config.lr, weight_decay=config.weight_decay)

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.T_max, config.eta_min)

    def batch_training(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        # forward
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        acc = accuracy(outputs, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))
        self.average_meters['loss'].update(loss.item(), images.size(0))

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        logits = self.model(images)
        acc = accuracy(logits, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))


if __name__ == '__main__':
    trainer = BaselineTrainer()
    # print(trainer.model)
    trainer.train()
