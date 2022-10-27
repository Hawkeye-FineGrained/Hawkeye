import os
import sys
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from shutil import copyfile

from config import setup_config
from dataset.dataset import FGDataset
from dataset.transforms import ClassificationPresetTrain, ClassificationPresetEval
from model.registry import MODEL
from utils import PerformanceMeter, TqdmHandler, set_random_seed, AverageMeter, accuracy, Timer


def emergency_save(func):
    """ Save checkpoint when `KeyboardInterrupt` or other errors occur.
    """

    def _emergency_save(self):
        try:
            func(self)
        except KeyboardInterrupt:
            self.logger.info('KeyboardInterrupt - try to save checkpoint ...')
            self.save_checkpoint()
        except Exception as e:
            import traceback
            self.logger.error(repr(e))
            self.logger.error(traceback.format_exc())
            self.logger.info('try to save checkpoint ...')
            self.save_checkpoint()

    return _emergency_save


class Trainer(object):
    """Base trainer
    """

    def __init__(self):
        self.config = setup_config()

        # set epoch, resume flag and log_root
        self.epoch = 0
        self.start_epoch = 0
        self.total_epoch = self.config.train.epoch
        self.resume = 'resume' in self.config.experiment and self.config.experiment.resume
        self.debug = self.config.experiment.debug if 'debug' in self.config.experiment else False
        self.log_root = os.path.join(self.config.experiment.log_dir, self.config.experiment.name)
        self.report_one_line = True  # logger report acc and loss in one line when training

        # log root directory should not already exist
        if not self.resume and not self.debug:
            assert not os.path.exists(self.log_root), 'Experiment log folder already exists!!'
            # create log root directory and copy
            os.makedirs(self.log_root)
            print(f'Created log directory: {self.log_root}')
            # copy yaml file and train.py
            with open(os.path.join(self.log_root, 'train_config.yaml'), 'w') as f:
                f.write(self.config.__str__())
            copyfile(sys.argv[0], os.path.join(self.log_root, 'train.py'))

        # logger and tensorboard writer
        self.logger = self.get_logger()
        self.tb_writer = SummaryWriter(self.log_root)
        self.logger.info(f'Train Config:\n{self.config.__str__()}')

        # set device. `config.experiment.cuda` should be a list of gpu device ids, None or [] for cpu only.
        self.device = self.config.experiment.cuda if isinstance(self.config.experiment.cuda, list) else []
        if len(self.device) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.device])
            self.logger.info(f'Using GPU: {self.device}')
        else:
            self.logger.info(f'Using CPU！')

        # set random seed
        if 'seed' in self.config.experiment and self.config.experiment.seed is not None:
            set_random_seed(self.config.experiment.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.logger.info(f'Using specific random seed: {self.config.experiment.seed}')

        # build dataloader and model
        self.transformers = self.get_transformers(self.config.dataset.transformer)
        self.collate_fn = self.get_collate_fn()
        self.datasets = self.get_dataset(self.config.dataset)
        self.dataloaders = self.get_dataloader(self.config.dataset)
        self.logger.info(f'Building model {self.config.model.name} ...')
        self.model = self.get_model(self.config.model)
        self.model = self.to_device(self.model, parallel=True)
        self.logger.info(f'Building model {self.config.model.name} OK!')

        self.criterion = self.get_criterion(self.config.train.criterion)
        self.optimizer = self.get_optimizer(self.config.train.optimizer)
        self.scheduler = self.get_scheduler(self.config.train.scheduler)

        # resume from checkpoint
        if self.resume:
            self.logger.info(f'Resuming from `{self.resume}`')
            self.load_checkpoint(self.config.experiment.resume)

        # build meters
        self.performance_meters = self.get_performance_meters()
        self.average_meters = self.get_average_meters()

        # timer
        self.timer = Timer()

        self.logger.info('Training Preparation Done!')

    def __del__(self):
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()

    def get_logger(self):
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.INFO)

        screen_handler = TqdmHandler()
        screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(screen_handler)

        complicated_format = logging.Formatter('%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s \
                            %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        simple_format = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

        file_handler = logging.FileHandler(os.path.join(self.log_root, 'report.log'), encoding='utf8')
        file_handler.setFormatter(simple_format)
        logger.addHandler(file_handler)
        return logger

    def get_performance_meters(self):
        return {
            'train': {
                metric: PerformanceMeter(higher_is_better=False if 'loss' in metric else True)
                for metric in ['acc', 'loss']
            },
            'val': {
                metric: PerformanceMeter() for metric in ['acc']
            },
            'val_first': {
                metric: PerformanceMeter() for metric in ['acc']
            }
        }

    def get_average_meters(self):
        meters = ['acc', 'loss']  # Reset every epoch. 'acc' is reused in train/val/val_first stage.
        return {
            meter: AverageMeter() for meter in meters
        }

    def reset_average_meters(self):
        for meter in self.average_meters:
            self.average_meters[meter].reset()

    def get_model(self, config):
        """Build and load model in config
        """
        name = config.name
        model = MODEL.get(name)(config)

        if 'load' in config and config.load != '':
            self.logger.info(f'Loading model from {config.load}')
            state_dict = torch.load(config.load, map_location='cpu')
            model.load_state_dict(state_dict)
            self.logger.info(f'OK! Model loaded from {config.load}')
        return model

    def get_transformers(self, config):
        transformers = {
            'train': ClassificationPresetTrain(
                crop_size=config['image_size'],
                auto_augment_policy="ta_wide",
                random_erase_prob=0.1,
            ),
            'val': ClassificationPresetEval(
                crop_size=config['image_size'],
                resize_size=config['resize_size']
            )
        }
        return transformers

    def get_collate_fn(self):
        return {
            'train': None,
            'val': None
        }

    def get_dataset(self, config):
        splits = ['train', 'val']
        meta_paths = {
            split: os.path.join(config.meta_dir, split + '.txt') for split in splits
        }
        return {
            split: FGDataset(config.root_dir, meta_paths[split], transform=self.transformers[split]) for split in splits
        }

    def get_dataloader(self, config):
        splits = ['train', 'val']
        dataloaders = {
            split: DataLoader(
                self.datasets[split],
                config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=split == 'train',
                collate_fn=self.collate_fn[split]
            ) for split in splits
        }
        return dataloaders

    def get_criterion(self, config):
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def get_scheduler(self, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.T_max, config.eta_min)

    def to_device(self, m, parallel=False):
        if len(self.device) == 0:
            m = m.to('cpu')
        elif len(self.device) == 1 or not parallel:
            m = m.to(f'cuda:{self.device[0]}')
        else:
            m = m.cuda(self.device[0])
            m = torch.nn.DataParallel(m, device_ids=self.device)
        return m

    def get_model_module(self, model=None):
        """get `model` in single-gpu mode or `model.module` in multi-gpu mode.
        """
        if model is None:
            model = self.model
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        else:
            return model

    @emergency_save
    def train(self):
        config = self.config.train  # local config for training stage

        # validate firstly
        if 'val_first' in config and config.val_first:
            self.logger.info('Validate model before training.')
            self.validate()
            self.performance_meters['val_first']['acc'].update(self.average_meters['acc'].avg)
            self.report(epoch=0, split='val_first')

        self.model.train()

        for epoch in range(self.start_epoch, self.total_epoch):
            self.epoch = epoch
            self.reset_average_meters()
            self._on_start_epoch()

            # train stage
            self.logger.info(f'Starting epoch {epoch + 1} ...')
            self.timer.tick()
            training_bar = tqdm(self.dataloaders['train'], ncols=100)
            for data in training_bar:
                self._on_start_forward()
                self.batch_training(data)
                self._on_end_forward()
                training_bar.set_description(f'Train Epoch [{self.epoch + 1}/{self.total_epoch}]')
                training_bar.set_postfix(acc=self.average_meters['acc'].avg, loss=self.average_meters['loss'].avg)
            duration = self.timer.tick()
            self.logger.info(f'Training duration {duration:.2f}s!')

            # train stage metrics
            self.update_performance_meter('train')
            self.report(epoch=epoch + 1, split='train')

            # val stage
            self.logger.info(f'Starting validation stage in epoch {epoch + 1} ...')
            self.timer.tick()
            # validate
            self.validate()
            duration = self.timer.tick()
            self.logger.info(f'Validation duration {duration:.2f}s!')

            # val stage metrics
            val_acc = self.average_meters['acc'].avg
            if self.performance_meters['val']['acc'].best_value is not None:
                is_best = epoch >= 5 and val_acc > self.performance_meters['val']['acc'].best_value
            else:
                is_best = epoch >= 5
            self.update_performance_meter('val')
            self.report(epoch=epoch + 1, split='val')

            self.do_scheduler_step()
            self.logger.info(f'Epoch {epoch + 1} Done!')

            # save model
            if epoch != 0 and (epoch + 1) % config.save_frequence == 0:
                self.logger.info('Saving model ...')
                self.save_model()
                # self.logger.info('Saving checkpoint ...')
                # self.save_checkpoint()
            if is_best:
                self.logger.info('Saving best model ...')
                self.save_model('best_model.pth')

            # hook: on_end_epoch
            self._on_end_epoch()

        self.logger.info(f'best acc:{self.performance_meters["val"]["acc"].best_value}')

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

    def validate(self):
        self.model.train(False)
        self.reset_average_meters()

        with torch.no_grad():
            val_bar = tqdm(self.dataloaders['val'], ncols=100)
            for data in val_bar:
                self.batch_validate(data)
                val_bar.set_description(f'Val Epoch [{self.epoch + 1}/{self.total_epoch}]')
                val_bar.set_postfix(acc=self.average_meters['acc'].avg)

        self.model.train(True)

    def batch_validate(self, data):
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        logits = self.model(images)
        acc = accuracy(logits, labels, 1)
        self.average_meters['acc'].update(acc, images.size(0))

    def do_scheduler_step(self):
        self.scheduler.step()

    def update_performance_meter(self, split):
        if split == 'train':
            self.performance_meters['train']['acc'].update(self.average_meters['acc'].avg)
            self.performance_meters['train']['loss'].update(self.average_meters['loss'].avg)
        elif split == 'val':
            self.performance_meters['val']['acc'].update(self.average_meters['acc'].avg)

    def report(self, epoch, split='train'):
        # tensorboard summary-writer and logger
        for metric in self.performance_meters[split]:
            value = self.performance_meters[split][metric].current_value
            self.tb_writer.add_scalar(f'{split}/{metric}', value, global_step=epoch)
            if not self.report_one_line:
                self.logger.info(f'Epoch:{epoch}\t{split}/{metric}: {value}')
        if self.report_one_line:
            metric_str = '  '.join([f'{metric}: {self.performance_meters[split][metric].current_value:.2f}'
                                    for metric in self.performance_meters[split]])
            self.logger.info(f'Epoch:{epoch}\t{metric_str}')

    def save_model(self, name=None):
        model_name = self.config.model.name
        if name is None:
            path = os.path.join(self.log_root, f'{model_name}_epoch_{self.epoch + 1}.pth')
        else:
            path = os.path.join(self.log_root, name)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f'model saved to: {path}')

    def save_checkpoint(self):
        path = os.path.join(self.log_root, f'checkpoint_epoch_{self.epoch}.pth')
        checkpoint = {
            'epoch': self.epoch,
            'model': self.get_model_module().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, path)
        self.logger.info(f'checkpoint successfully saved to: {path}')

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.logger.info(f'load checkpoint from: {path}, start_epoch: {checkpoint["epoch"]}')

    # hooks used in trainer
    def _on_start_epoch(self):
        if 'hook' in self.config and 'on_start_epoch' in self.config.hook:
            return self.on_start_epoch(self.config.hook.on_start_epoch)
        else:
            return self.on_start_epoch(None)

    def _on_end_epoch(self):
        if 'hook' in self.config and 'on_end_epoch' in self.config.hook:
            return self.on_end_epoch(self.config.hook.on_end_epoch)
        else:
            return self.on_end_epoch(None)

    def _on_start_forward(self):
        if 'hook' in self.config and 'on_start_forward' in self.config.hook:
            return self.on_start_forward(self.config.hook.on_start_forward)
        else:
            return self.on_start_forward(None)

    def _on_end_forward(self):
        if 'hook' in self.config and 'on_end_forward' in self.config.hook:
            return self.on_end_forward(self.config.hook.on_end_forward)
        else:
            return self.on_end_forward(None)

    # hooks to implement
    def on_start_epoch(self, config):
        lrs_str = "  ".join([f'{p["lr"]}' for p in self.optimizer.param_groups])
        self.logger.info(f'Epoch:{self.epoch + 1}  LR: {lrs_str}')

    def on_end_epoch(self, config):
        pass

    def on_start_forward(self, config):
        pass

    def on_end_forward(self, config):
        pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
