import logging
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN


class PerformanceMeter(object):
    """Record the performance metric during training
    """

    def __init__(self, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        self.current_value = self.values[-1]
        self.best_value = self.best_function(self.values)
        self.best_epoch = self.values.index(self.best_value)

    @property
    def value(self):
        return self.values[-1]


class AverageMeter(object):
    """Keep track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        super(TqdmHandler, self).__init__()

    def emit(self, msg):
        msg = self.format(msg)
        tqdm.write(msg)
        time.sleep(1)


class Timer(object):

    def __init__(self):
        self.start = time.time()
        self.last = time.time()

    def tick(self, from_start=False):
        this_time = time.time()
        if from_start:
            duration = this_time - self.start
        else:
            duration = this_time - self.last
        self.last = this_time
        return duration


def build_config_from_dict(_dict):
    cfg = CN()
    for key in _dict:
        cfg[key] = _dict[key]
    return cfg


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
