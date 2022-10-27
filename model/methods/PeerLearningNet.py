import copy
import torch.nn as nn

from model.registry import MODEL
from model.utils import initialize_weights


@MODEL.register
class PeerLearningNet(nn.Module):

    def __init__(self, config):
        super(PeerLearningNet, self).__init__()
        self.base_model = MODEL.get(config.base_model.name)(config.base_model)
        self.base_model2 = copy.deepcopy(self.base_model)
        self.base_model2.classifier.apply(initialize_weights)

    def forward(self, x):
        out1 = self.base_model(x)
        out2 = self.base_model2(x)
        return out1, out2
