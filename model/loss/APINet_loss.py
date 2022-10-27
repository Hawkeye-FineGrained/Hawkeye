import torch
from torch import nn


class APINetLoss(nn.Module):
    def __init__(self, config):
        super(APINetLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.rank_loss = nn.MarginRankingLoss(margin=0.05)
        self.softmax_layer = nn.Softmax(dim=1)

    def __call__(self, output, target):
        # logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2 = output
        self_logits, other_logits, labels1, labels2 = output
        device = labels1.device

        batch_size = self_logits.shape[0] // 2
        # labels1 = labels1.to(self._device)
        # labels2 = labels2.to(self._device)

        # self_logits = torch.zeros(2 * batch_size, 200).to(device)
        # other_logits = torch.zeros(2 * batch_size, 200).to(device)
        # self_logits[:batch_size] = logit1_self
        # self_logits[batch_size:] = logit2_self
        # other_logits[:batch_size] = logit1_other
        # other_logits[batch_size:] = logit2_other

        # compute loss
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        softmax_loss = self.ce_loss(logits, targets)

        self_scores = self.softmax_layer(self_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                      torch.cat([labels1, labels2], dim=0)]
        other_scores = self.softmax_layer(other_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                        torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones((2 * batch_size, )).to(device)
        rank_loss = self.rank_loss(self_scores, other_scores, flag)
        loss = softmax_loss + rank_loss
        return loss
