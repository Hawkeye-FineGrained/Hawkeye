import os
import sys
import torch
from torchvision.transforms import transforms

sys.path.append(os.path.abspath('.'))
from model.loss.DCL_loss import DCLLoss
from dataset.transforms import RandomSwap
from dataset.dataset_DCL import DCLDataset, collate_fn4train, collate_fn4val
from train import Trainer
from utils import accuracy


class BaselineTrainer(Trainer):
    def __init__(self):
        super(BaselineTrainer, self).__init__()
        self.num_classes = self.config.model.num_classes

    def get_transformers(self, config):
        resize_reso = config.resize_size if 'resize_size' in config else 512
        crop_reso = config.image_size if 'image_size' in config else 448
        swap_num = config.swap_num if 'swap_num' in config else [7, 7]

        return {
            'swap': transforms.Compose([
                RandomSwap((swap_num[0], swap_num[1])),
            ]),
            'common_aug': transforms.Compose([
                transforms.Resize((resize_reso, resize_reso)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop((crop_reso, crop_reso)),
                transforms.RandomHorizontalFlip(),
            ]),
            'train_totensor': transforms.Compose([
                transforms.Resize((crop_reso, crop_reso)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val_totensor': transforms.Compose([
                transforms.Resize((crop_reso, crop_reso)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'test_totensor': transforms.Compose([
                transforms.Resize((resize_reso, resize_reso)),
                transforms.CenterCrop((crop_reso, crop_reso)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'None': None,
        }

    def get_collate_fn(self):
        return {
            'train': collate_fn4train,
            'val': collate_fn4val
        }

    def get_dataset(self, config):
        splits = ['train', 'val']
        meta_paths = {
            split: os.path.join(config.meta_dir, split + '.txt') for split in splits
        }
        return {
            split: DCLDataset(
                config.root_dir, meta_paths[split], transforms=self.transformers, mode=split,
                cls_2=self.config.model.cls_2, cls_2xmul=self.config.model.cls_2xmul
            ) for split in splits
        }

    def get_criterion(self, config):
        return DCLLoss(config)

    def get_optimizer(self, config):
        ignored_params1 = list(map(id, self.get_model_module().classifier.parameters()))
        ignored_params2 = list(map(id, self.get_model_module().classifier_swap.parameters()))
        ignored_params3 = list(map(id, self.get_model_module().Convmask.parameters()))
        ignored_params = ignored_params1 + ignored_params2 + ignored_params3
        base_params = filter(lambda p: id(p) not in ignored_params, self.get_model_module().parameters())

        classifier_lr = config.lr_ratio * config.lr
        optimizer = torch.optim.SGD([
            {'params': base_params, 'lr': config.lr},
            {'params': self.get_model_module().classifier.parameters(), 'lr': classifier_lr},
            {'params': self.get_model_module().classifier_swap.parameters(), 'lr': classifier_lr},
            {'params': self.get_model_module().Convmask.parameters(), 'lr': classifier_lr},
        ], momentum=config.momentum)
        return optimizer

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)

    def batch_training(self, data):
        inputs, labels, labels_swap, swap_law, img_names = data

        inputs = self.to_device(inputs)
        labels = self.to_device(labels)
        labels_swap = self.to_device(labels_swap)
        swap_law = self.to_device(swap_law)

        # forward
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels, labels_swap, swap_law)
        if self.config.model.cls_2xmul:
            logit = outputs[0] + outputs[1][:, 0:self.num_classes] + outputs[1][:, self.num_classes:2 * self.num_classes]
        else:
            logit = outputs[0]
        acc = accuracy(logit, labels, 1)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        self.average_meters['acc'].update(acc, labels.size(0))
        self.average_meters['loss'].update(loss.item(), labels.size(0))

    def batch_validate(self, data):
        inputs = self.to_device(data[0])
        labels = self.to_device(data[1].long())

        # forward
        outputs = self.model(inputs)
        if self.config.model.cls_2xmul:
            logit = outputs[0] + outputs[1][:, 0:self.num_classes] + outputs[1][:, self.num_classes:2 * self.num_classes]
        else:
            logit = outputs[0]
        acc = accuracy(logit, labels, 1)

        self.average_meters['acc'].update(acc, labels.size(0))


if __name__ == '__main__':
    trainer = BaselineTrainer()
    # print(trainer.model)
    trainer.train()
