import os
import sys
import torch
from torchvision import transforms

sys.path.append(os.path.abspath('.'))
from train import Trainer


class BCNNTrainer(Trainer):
    def __init__(self):
        super(BCNNTrainer, self).__init__()

    # def get_transformers(self, config):
    #     transformers = {
    #         'train': transforms.Compose([
    #             transforms.Resize(size=config.image_size),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomCrop(size=config.image_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         ]),
    #         'val': transforms.Compose([
    #             transforms.Resize(size=config.image_size),
    #             transforms.CenterCrop(size=config.image_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         ])
    #     }
    #     return transformers

    def get_optimizer(self, config):
        model = self.get_model_module()
        # Stage 1, freeze backbone parameters.
        if self.config.model.stage == 1:
            params = model.classifier.parameters()
        # Stage 2, train all parameters.
        elif self.config.model.stage == 2:
            params = model.parameters()
        return torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)

    def do_scheduler_step(self):
        metric = self.performance_meters['val']['acc'].value
        self.scheduler.step(metric)


if __name__ == '__main__':
    trainer = BCNNTrainer()
    # print(trainer.model)
    trainer.train()
