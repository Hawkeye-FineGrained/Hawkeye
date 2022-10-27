import torch
from torch import nn
from model.backbone import vgg16
from model.utils import initialize_weights
from model.registry import MODEL


class BilinearPooling(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, x):
        batch_size = x.size(0)
        channel_size = x.size(1)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, channel_size, feature_size)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size

        x = x.view(batch_size, -1)
        x = torch.sqrt(x + 1e-5)

        # x = x.view(batch_size, -1)
        # x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)

        x = torch.nn.functional.normalize(x)
        return x


@MODEL.register
class BCNN(nn.Module):

    def __init__(self, config):
        super(BCNN, self).__init__()
        # Training stage for BCNN. Stage 1 freeze backbone parameters.
        self.stage = config.stage if 'stage' in config else 2

        self.backbone = vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2][0])

        self.bilinear_pooling = BilinearPooling()
        self.classifier = nn.Linear(512 ** 2, config.num_classes)
        self.classifier.apply(initialize_weights)

        if self.stage == 1:
            for params in self.backbone.parameters():
                params.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        if self.stage == 1:
            x = x.detach()
        x = self.bilinear_pooling(x)
        x = self.classifier(x)
        return x
