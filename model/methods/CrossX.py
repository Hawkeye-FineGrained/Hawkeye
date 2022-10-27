import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F
from model.registry import MODEL

eps = np.finfo(float).eps
device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

__all__ = ['ResNet', 'resnet50', 'CrossX']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MELayer(nn.Module):
    def __init__(self, channel, reduction=16, nparts=1):
        super(MELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nparts = nparts
        parts = list()
        for part in range(self.nparts):
            parts.append(nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
            ))
        self.parts = nn.Sequential(*parts)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        meouts = list()
        for i in range(self.nparts):
            meouts.append(x * self.parts[i](y).view(b, c, 1, 1))

        return meouts


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, meflag=False, nparts=1, reduction=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.meflag = meflag
        if self.meflag:
            self.me = MELayer(planes * 4, nparts=nparts, reduction=reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.meflag:

            outreach = out.clone()
            parts = self.me(outreach)

            out += residual
            out = self.relu(out)

            for i in range(len(parts)):
                parts[i] = self.relu(parts[i] + residual)
            return out, parts
        else:
            out += residual
            out = self.relu(out)
            return out


class ResNet(nn.Module):

    def __init__(self, block, layers, nparts=1, meflag=False, num_classes=1000):
        self.nparts = nparts
        self.nclass = num_classes
        self.meflag = meflag
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], meflag=meflag, stride=2, nparts=nparts, reduction=256)
        self.layer4 = self._make_layer(block, 512, layers[3], meflag=meflag, stride=2, nparts=nparts, reduction=256)
        self.adpavgpool = nn.AdaptiveAvgPool2d(1)

        # if meflag == False, vanilla resnet
        self.fc_ulti = nn.Linear(512 * block.expansion * nparts, num_classes)

        # if meflag == True, multiple branch outputs
        if self.nparts > 1:
            self.adpmaxpool = nn.AdaptiveMaxPool2d(1)
            self.fc_plty = nn.Linear(256 * block.expansion * nparts, num_classes)
            self.fc_cmbn = nn.Linear(256 * block.expansion * nparts, num_classes)

            # dimension reducing for the last convolutional layer
            self.conv2_1 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
            self.conv2_2 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)

            # combinign feature maps from the penultimate layer and the dimension-reduced final layer
            self.conv3_1 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
            self.conv3_2 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
            self.bn3_1 = nn.BatchNorm2d(256 * block.expansion)
            self.bn3_2 = nn.BatchNorm2d(256 * block.expansion)

            if nparts == 3:
                self.conv2_3 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
                self.conv3_3 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1,
                                         bias=False)
                self.bn3_3 = nn.BatchNorm2d(256 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, meflag=False, stride=1, nparts=1, reduction=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, meflag=meflag, nparts=nparts, reduction=reduction))
            else:
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.meflag:
            x, plty_parts = self.layer3(x)
            _, ulti_parts = self.layer4(x)

            cmbn_ftres = list()
            for i in range(self.nparts):
                # pdb.set_trace()
                if i == 0:
                    ulti_parts_iplt = F.interpolate(self.conv2_1(ulti_parts[i]), 28)
                    cmbn_ftres.append(
                        self.adpavgpool(self.bn3_1(self.conv3_1(torch.add(plty_parts[i], ulti_parts_iplt)))))
                elif i == 1:
                    ulti_parts_iplt = F.interpolate(self.conv2_2(ulti_parts[i]), 28)
                    cmbn_ftres.append(
                        self.adpavgpool(self.bn3_2(self.conv3_2(torch.add(plty_parts[i], ulti_parts_iplt)))))
                elif i == 2:
                    ulti_parts_iplt = F.interpolate(self.conv2_3(ulti_parts[i]), 28)
                    cmbn_ftres.append(
                        self.adpavgpool(self.bn3_3(self.conv3_3(torch.add(plty_parts[i], ulti_parts_iplt)))))

                plty_parts[i] = self.adpmaxpool(plty_parts[i])
                ulti_parts[i] = self.adpavgpool(ulti_parts[i])

            # for the penultimate layer
            # pdb.set_trace()
            xp = torch.cat(plty_parts, 1)
            xp = xp.view(xp.size(0), -1)
            xp = self.fc_plty(xp)

            # for the final layer
            xf = torch.cat(ulti_parts, 1)
            xf = xf.view(xf.size(0), -1)
            xf = self.fc_ulti(xf)

            # for the combined feature
            xc = torch.cat(cmbn_ftres, 1)
            xc = xc.view(xc.size(0), -1)
            xc = self.fc_cmbn(xc)

            return xf, xp, xc, ulti_parts, plty_parts, cmbn_ftres

        else:
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.adpavgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_ulti(x)

            return x


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


@MODEL.register
def CrossX(config):
    pretrained = config.pretrained if 'pretrained' in config else True
    meflag = True if config.num_parts > 1 else False    # meflag==True: resnet with osme
    num_classes = config.num_classes if 'num_classes' in config else 200
    model = resnet50(nparts=config.num_parts, meflag=meflag, num_classes=num_classes, pretrained=pretrained)
    return model
