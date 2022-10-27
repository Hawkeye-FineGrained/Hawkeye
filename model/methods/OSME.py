import torch
from torch import nn

from model.backbone import resnet101
from model.registry import MODEL


class OSME_block(torch.nn.Module):
    def __init__(self, channels, ratio):
        torch.nn.Module.__init__(self)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _, _, = x.size()
        z = self.avg_pool(x).squeeze()
        m = self.block(z)
        s = m.view(N, C, 1, 1) * x
        return s


class OSME(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1024, feature_shape=(7, 7), num_attention=2):
        torch.nn.Module.__init__(self)
        reduce_ratio = 16
        fc_in_channels = in_channels * feature_shape[0] * feature_shape[1] if isinstance(feature_shape, tuple) \
            else in_channels * feature_shape * feature_shape
        self.blocks = nn.ModuleList([OSME_block(in_channels, reduce_ratio) for _ in range(num_attention)])
        self.fcs = nn.ModuleList([nn.Linear(fc_in_channels, out_channels) for _ in range(num_attention)])

    def forward(self, x):
        """
        :param x: [N D W H]
        :return: [N C], [N P C]
        """
        N, C, _, _ = x.size()
        s = [block(x) for block in self.blocks]
        features = [fc(s[i].view(N, -1)) for i, fc in enumerate(self.fcs)]
        return sum(features), torch.stack(features, dim=1)


@MODEL.register
class OSMENet(nn.Module):
    def __init__(self, config):
        super(OSMENet, self).__init__()
        self.config = config
        self.num_attention = config.num_attention   # Number of attention regions (`P` in paper).
        self.num_classes = config.num_classes

        resnet = resnet101(pretrained=True)  # resnet101
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.osme = OSME(2048, 1024, feature_shape=7, num_attention=self.num_attention)
        self.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.backbone(x)  # N 2048 7 7
        x1, x_part = self.osme(x)  # [N 1024] [N 2 1024]
        out = self.classifier(x1)  # [N 200]
        return out, x_part
