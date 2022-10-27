import torch
import torch.nn.functional as F
from torch import nn


class MultiSmoothLoss(nn.Module):
    """Multi smooth loss.
    """

    def __init__(self, config):
        self.smooth_ratio = config.smooth_ratio

    def __call__(self, output, target,
                 loss_weight=None, weight=None, size_average=True, ignore_index=-100, reduce=True):
        assert isinstance(output, tuple), 'input is less than 2'

        weight_loss = torch.ones(len(output)).to(output[0].device)
        if loss_weight is not None:
            for item in loss_weight.items():
                weight_loss[int(item[0])] = item[1]

        loss = 0
        for i in range(0, len(output)):
            if i in [1, len(output) - 1]:
                prob = F.log_softmax(output[i], dim=1)
                ymask = prob.data.new(prob.size()).zero_()
                ymask = ymask.scatter_(1, target.view(-1, 1), 1)
                ymask = self.smooth_ratio * ymask + (1 - self.smooth_ratio) * (1 - ymask) / (output[i].shape[1] - 1)
                loss_tmp = - weight_loss[i] * ((prob * ymask).sum(1).mean())
            else:
                loss_tmp = weight_loss[i] * F.cross_entropy(output[i], target, weight, ignore_index=ignore_index,
                                                            reduction='mean')
            loss += loss_tmp

        return loss
