import torch
import torch.nn.functional as F


def PeerLearningLoss(logits_1, logits_2, labels, drop_rate):
    """peer learning loss
    Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach. [ICCV 2021]
    :param logits_1:    shape of (N, 200)
    :param logits_2:    shape of (N, 200)
    :param labels:      shape of (N,)
    :param drop_rate:   drop_rate for each epoch
    :return:
    """
    dist_1 = F.softmax(logits_1, dim=1)
    dist_2 = F.softmax(logits_2, dim=1)

    _, pred_1 = dist_1.topk(1, dim=1, largest=True, sorted=True)  # (N, 1)
    _, pred_2 = dist_2.topk(1, dim=1, largest=True, sorted=True)  # (N, 1)
    pred_1 = pred_1.squeeze(dim=1)
    pred_2 = pred_2.squeeze(dim=1)

    disagreement_index = (pred_1 != pred_2).nonzero().squeeze(dim=1)  # (n)
    agreement_index = (pred_1 == pred_2).nonzero().squeeze(dim=1)  # (N-n)

    logits_1_disagree = logits_1[disagreement_index]
    logits_2_disagree = logits_2[disagreement_index]
    labels_disagree = labels[disagreement_index]

    logits_1_agree = logits_1[agreement_index]
    logits_2_agree = logits_2[agreement_index]
    labels_agree = labels[agreement_index]

    if agreement_index.shape[0] > 0:
        loss_1_agree = F.cross_entropy(logits_1_agree, labels_agree,
                                       reduction='none')  # (N) loss per instance in this batch
        ind_1_sorted = torch.argsort(loss_1_agree.data)  # (N) sorted index of the loss

        loss_2_agree = F.cross_entropy(logits_2_agree, labels_agree, reduction='none')
        ind_2_sorted = torch.argsort(loss_2_agree.data)

        num_remember = int((1 - drop_rate) * loss_1_agree.shape[0])

        ind_1_update = ind_1_sorted[:num_remember]  # select the first num_remember low-loss instances
        ind_2_update = ind_2_sorted[:num_remember]

        if disagreement_index.shape[0] > 0:
            logits_1_final = torch.cat((logits_1_disagree, logits_1_agree[ind_2_update]), dim=0)
            labels_1_final = torch.cat((labels_disagree, labels_agree[ind_2_update]), dim=0)
            logits_2_final = torch.cat((logits_2_disagree, logits_2_agree[ind_1_update]), dim=0)
            labels_2_final = torch.cat((labels_disagree, labels_agree[ind_1_update]), dim=0)
        else:
            logits_1_final = logits_1_agree[ind_2_update]
            labels_1_final = labels_agree[ind_2_update]
            logits_2_final = logits_2_agree[ind_1_update]
            labels_2_final = labels_agree[ind_1_update]
    else:
        logits_1_final = logits_1_disagree
        labels_1_final = labels_disagree
        logits_2_final = logits_2_disagree
        labels_2_final = labels_disagree

    loss_1_update = F.cross_entropy(logits_1_final, labels_1_final)
    loss_2_update = F.cross_entropy(logits_2_final, labels_2_final)

    return loss_1_update, loss_2_update
