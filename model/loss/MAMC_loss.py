import torch
import torch.nn as nn
import torch.nn.functional as F


class MAMCLoss(nn.Module):
    def __init__(self, config):
        super(MAMCLoss, self).__init__()
        self.lambda_a = config.lambda_a if 'lambda_a' in config else 0.5    # The coefficient of n-pair loss.
        self.use_mamc = config.use_mamc if 'use_mamc' in config else True   # Whether use MAMC loss or not.
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.npair_loss = NPairsLoss()

    def forward(self, inputs, targets):
        pred, x_part = inputs
        loss_ce = self.ce_loss(pred, targets)
        if not self.use_mamc:
            return loss_ce

        loss_npair = self.npair_loss(x_part, targets)
        return loss_ce + self.lambda_a * loss_npair


class NPairsLoss(nn.Module):
    """N-pairs loss as explained in equation 11 of MAMC paper.

    Reference:
        Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition
    """

    def __init__(self):
        super(NPairsLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, part_num, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        b, p, _ = inputs.size()
        n = b * p
        inputs = inputs.contiguous().view(n, -1)
        inputs = F.normalize(inputs, p=2, dim=1)
        targets = torch.repeat_interleave(targets, p)
        parts = torch.arange(p).repeat(b)
        prod = torch.mm(inputs, inputs.t())

        parts = parts.to(inputs.device)

        same_class_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        same_atten_mask = parts.expand(n, n).eq(parts.expand(n, n).t())

        s_sasc = same_class_mask & same_atten_mask
        s_sadc = (~same_class_mask) & same_atten_mask
        s_dasc = same_class_mask & (~same_atten_mask)
        s_dadc = (~same_class_mask) & (~same_atten_mask)

        # For each anchor, compute equation (11) of paper
        loss_sasc = 0
        loss_sadc = 0
        loss_dasc = 0
        for i in range(n):
            # loss_sasc
            pos = prod[i][s_sasc[i]]
            neg = prod[i][s_sadc[i] | s_dasc[i] | s_dadc[i]]
            n_pos = pos.size(0)
            n_neg = neg.size(0)
            pos = pos.repeat(n_neg, 1).t()
            neg = neg.repeat(n_pos, 1)
            loss_sasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

            # loss_sadc
            pos = prod[i][s_sadc[i]]
            neg = prod[i][s_dadc[i]]
            n_pos = pos.size(0)
            n_neg = neg.size(0)
            pos = pos.repeat(n_neg, 1).t()
            neg = neg.repeat(n_pos, 1)
            loss_sadc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

            # loss_dasc
            pos = prod[i][s_dasc[i]]
            neg = prod[i][s_dadc[i]]
            n_pos = pos.size(0)
            n_neg = neg.size(0)
            pos = pos.repeat(n_neg, 1).t()
            neg = neg.repeat(n_pos, 1)
            loss_dasc += torch.sum(torch.log(1 + torch.sum(torch.exp(neg - pos), dim=1)))

        return (loss_sasc + loss_sadc + loss_dasc) / n
