import torch
from torch import nn
import torch.nn.functional as F

from model.backbone import resnet50
from model.registry import MODEL
from model.utils import initialize_weights


class ChannelInteractionModule(nn.Module):
    """Channel Interaction Network
    """

    def __init__(self, in_channel=2048, spatial_size=(7, 7)):
        super(ChannelInteractionModule, self).__init__()
        self.in_channel = in_channel
        self.spatial_size = spatial_size
        WH = self.spatial_size[0] * self.spatial_size[1]

        self.softmax = nn.Softmax()
        self.conv = nn.Conv2d(self.in_channel, self.in_channel, 3, 1, 1)
        self.fc = nn.Linear(2 * self.in_channel * WH, 1)

    def forward(self, x):
        # x -> B C W H
        B, C, W, H = x.size()
        assert B % 2 == 0, 'batch size should not be odd!'
        x = x.view(B, C, W * H)  # x -> B C WH

        # SCI Module
        bilinear_matrix = torch.bmm(x, torch.transpose(x, 1, 2)) / (W * H)  # B C C
        W_SCI = F.softmax(-bilinear_matrix, dim=2)  # B C C

        Y = torch.bmm(W_SCI, x)  # Y -> B C WH

        Y = self.conv(Y.view(B, C, W, H))
        Y = Y.view(B, C, W * H)
        Z = Y + x

        if not self.training:
            return Z

        # CCI Module
        y = Y.view(B, -1)
        y_a = torch.cat((y[:B // 2], y[B // 2:]), dim=1)
        y_b = torch.cat((y[B // 2:], y[:B // 2]), dim=1)
        eta = self.fc(y_a)
        gamma = self.fc(y_b)

        weight = torch.cat((eta, gamma), dim=0)
        W_SCI_BA = torch.cat((W_SCI[B // 2:], W_SCI[:B // 2]), dim=0)
        W_CCI = torch.abs(W_SCI - weight.view(-1, 1, 1) * W_SCI_BA)

        Y_CCI = torch.bmm(W_CCI, x)  # Y -> B C WH

        Y_CCI = self.conv(Y_CCI.view(B, C, W, H))
        Y_CCI = Y_CCI.view(B, C, W * H)
        Z_CCI = Y_CCI + x

        return Z, Z_CCI


class CINClassifier(nn.Module):
    """Channel Interaction Network Classifier
    """

    def __init__(self, in_channel=2048, num_classes=200):
        super(CINClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channel, num_classes)

    def forward(self, x):
        if isinstance(x, tuple):
            Z, Z_CCI = x
            Z = torch.squeeze(self.pool(Z))
            Z = self.classifier(Z)
            return Z, Z_CCI
        else:
            x = torch.squeeze(self.pool(x))
            x = self.classifier(x)
            return x


@MODEL.register
class CIN(nn.Module):
    def __init__(self, config):
        super(CIN, self).__init__()
        self.num_classes = config.num_classes if 'num_classes' in config else 200

        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.ChannelInteraction = ChannelInteractionModule(in_channel=2048, spatial_size=(7, 7))
        self.classifier = CINClassifier(in_channel=2048, num_classes=self.num_classes)

        self.ChannelInteraction.apply(initialize_weights)
        self.classifier.apply(initialize_weights)

    def forward(self, x):
        x = self.backbone(x)
        x = self.ChannelInteraction(x)
        x = self.classifier(x)
        return x
