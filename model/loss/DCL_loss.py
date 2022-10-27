from torch import nn


class DCLLoss(nn.Module):

    def __init__(self, config):
        super(DCLLoss, self).__init__()
        # Alpha, beta and gamma in Eq.(10), balancing different loss.
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.add_loss = nn.L1Loss()

    def __call__(self, outputs, labels, labels_swap, swap_law):
        loss_ce = self.ce_loss(outputs[0], labels)
        loss_swap = self.ce_loss(outputs[1], labels_swap)
        loss_law = self.add_loss(outputs[2], swap_law)
        loss = self.alpha * loss_ce + self.beta * loss_swap + self.gamma * loss_law
        return loss
