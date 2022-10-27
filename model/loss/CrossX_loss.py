import torch
from torch import nn
import torch.nn.functional as F


class RegularLoss(nn.Module):

    def __init__(self, gamma=0, num_parts=1):
        super(RegularLoss, self).__init__()
        self.num_parts = num_parts
        self.gamma = gamma

    def forward(self, x):
        assert isinstance(x, list), "parts features should be presented in a list"
        corr_matrix = torch.zeros(self.num_parts, self.num_parts)
        for i in range(self.num_parts):
            x[i] = x[i].squeeze()
            x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True))

        # original design
        for i in range(self.num_parts):
            for j in range(self.num_parts):
                corr_matrix[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    corr_matrix[i, j] = 1.0 - corr_matrix[i, j]
        regloss = torch.mul(torch.sum(torch.triu(corr_matrix)), self.gamma).to(x[0].device)

        return regloss


class CrossXLoss(nn.Module):

    def __init__(self, config):
        super(CrossXLoss, self).__init__()
        self.num_parts = config.num_parts   # Number of parts.
        self.gamma = config.gamma   # Gamma in Eq.(1) which balance different cost.

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ulti_loss = RegularLoss(gamma=self.gamma[0], num_parts=self.num_parts)
        self.plty_loss = RegularLoss(gamma=self.gamma[1], num_parts=self.num_parts)
        self.cmbn_loss = RegularLoss(gamma=self.gamma[2], num_parts=self.num_parts)
        self.kl_loss = nn.KLDivLoss(reduction='sum')

    def __call__(self, outputs, target):
        if self.num_parts == 1:
            loss = self.ce_loss(outputs, target)
        else:
            outputs_ulti, outputs_plty, outputs_cmbn, ulti_ftrs, plty_ftrs, cmbn_ftrs = outputs
            outs = outputs_ulti + outputs_plty + outputs_cmbn
            cls_loss = self.ce_loss(outs, target)

            reg_loss_cmbn = self.cmbn_loss(cmbn_ftrs)
            outputs_cmbn = F.log_softmax(outputs_cmbn, 1)

            reg_loss_ulti = self.ulti_loss(ulti_ftrs)
            reg_loss_plty = self.plty_loss(plty_ftrs)

            outputs_plty = F.log_softmax(outputs_plty, 1)
            outputs_ulti = F.softmax(outputs_ulti, 1)

            kl_loss = (self.kl_loss(outputs_plty, outputs_ulti) +
                       self.kl_loss(outputs_cmbn, outputs_ulti)) / target.size(0)
            loss = reg_loss_ulti + reg_loss_plty + reg_loss_cmbn + kl_loss + cls_loss
        return loss
