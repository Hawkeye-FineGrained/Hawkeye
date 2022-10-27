import torch
import torch.nn.functional as F
from torch import nn


class NTSLoss(nn.Module):

    def __init__(self, config):
        super(NTSLoss, self).__init__()
        self.PROPOSAL_NUM = config.proposal_num

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.list_loss = list_loss
        self.rank_loss = ranking_loss

    def __call__(self, outputs, targets):
        raw_logits, concat_logits, part_logits, _, top_n_prob = outputs
        batch_size = targets.size(0)

        part_loss = self.list_loss(part_logits.view(batch_size * self.PROPOSAL_NUM, -1),
                                   targets.unsqueeze(1).repeat(1, self.PROPOSAL_NUM).view(-1))
        part_loss = part_loss.view(batch_size, self.PROPOSAL_NUM)
        raw_loss = self.ce_loss(raw_logits, targets)
        concat_loss = self.ce_loss(concat_logits, targets)
        rank_loss = self.rank_loss(top_n_prob, part_loss, self.PROPOSAL_NUM)
        partcls_loss = self.ce_loss(part_logits.view(batch_size * self.PROPOSAL_NUM, -1),
                                    targets.unsqueeze(1).repeat(1, self.PROPOSAL_NUM).view(-1))
        loss = raw_loss + rank_loss + concat_loss + partcls_loss
        return loss


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=6):
    loss = torch.zeros(1).cuda()
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size
