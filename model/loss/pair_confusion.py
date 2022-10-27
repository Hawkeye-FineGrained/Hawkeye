"""Pairwise Confusion
    source: https://github.com/abhimanyudubey/confusion
"""
import torch
import torch.nn as nn


class PairwiseConfusionLoss(nn.Module):
    def __init__(self, config):
        super(PairwiseConfusionLoss, self).__init__()
        # Lambda, the coefficient of euclidean confusion loss.
        self.lambda_a = config.lambda_a if 'lambda_a' in config else 10
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, features, labels):
        batch_size = features.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = features[:int(0.5 * batch_size)]
        batch_right = features[int(0.5 * batch_size):]

        label_left = labels[:int(0.5 * batch_size)]
        label_right = labels[int(0.5 * batch_size):]

        loss = torch.norm((batch_left - batch_right).abs(), 2, 1)
        loss = loss * (label_left != label_right)
        loss = loss.sum() / float(batch_size)

        loss_ce = self.cross_entropy(features, labels)
        return loss_ce + self.lambda_a * loss


def EntropicConfusion(features):
    batch_size = features.size(0)
    return torch.mul(features, torch.log(features)).sum() * (1.0 / batch_size)
