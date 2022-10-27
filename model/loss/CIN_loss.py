import torch
from torch import nn

from model.utils import initialize_weights


class CINLoss(nn.Module):
    def __init__(self, config):
        super(CINLoss, self).__init__()
        # Alpha in Eq.(9). Weight of contrastive loss.
        self.alpha = config.alpha if 'alpha' in config else 2.0
        # Beta in E`.(8). A predefined margin in contrastive loss.
        self.beta = config.beta if 'beta' in config else 0.5
        # Channel of feature map `Z_CCI` which is the output of CCI module.
        self.channel = config.channel if 'channel' in config else 2048
        # Product of height and width in feature map `Z_CCI` which is the output of CCI module.
        self.feature_size = config.feature_size if 'feature_size' in config else 7 * 7
        # Output dimension of `h` in Eq.(8).
        self.r_channel = config.r_channel if 'r_channel' in config else 512

        self.pdist = nn.PairwiseDistance(p=2)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.h = nn.Linear(self.channel * self.feature_size, self.r_channel)
        self.apply(initialize_weights)

    def __call__(self, output, target):
        if not isinstance(output, tuple):
            return self.ce_loss(output, target)

        Z, Z_CCI = output
        B, C, WH = Z_CCI.size()

        # softmax loss
        loss_ce = self.ce_loss(Z, target)

        # contrastive loss
        Z_AB = Z_CCI.view(B, -1)
        Z_AB = self.h(Z_AB)
        pair_label = target[:B // 2] == target[B // 2]  # if y_ab = 1 or 0
        loss_cont_1 = torch.sum(torch.pow(self.pdist(Z_AB[:B // 2][pair_label], Z_AB[B // 2:][pair_label]), 2))
        loss_cont_2 = self.beta - self.pdist(Z_AB[:B // 2][~pair_label], Z_AB[B // 2:][~pair_label])
        loss_cont_2[loss_cont_2 < 0] = 0
        loss_cont_2 = torch.pow(loss_cont_1, 2)
        loss_cont = loss_cont_1 + loss_cont_2

        loss = loss_ce + self.alpha * loss_cont
        return loss
