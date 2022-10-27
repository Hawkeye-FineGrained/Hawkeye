import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from model.backbone import resnet50
from .grad_cam import GradCam
from ...registry import MODEL


def l2_norm_v2(input):
    input_size = input.size()
    _output = input / (torch.norm(input, p=2, dim=-1, keepdim=True))
    output = _output.view(input_size)
    return output


class Classifier(nn.Module):
    def __init__(self, in_panel, out_panel, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_panel, out_panel, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim() == 1:
            logit = logit.unsqueeze(0)
        return logit


def get_mask(conv5, layer_weights, mask=None, rate=0.8, img_size=448):
    mask_size = img_size // 32
    # conv5_cam = conv5.clone().detach().view(conv5.size(0), conv5.size(1), 14*14) * layer_weights.unsqueeze(-1)
    conv5_cam = conv5.clone().detach().view(conv5.size(0), conv5.size(1), -1) * layer_weights.unsqueeze(-1)

    mask_sum = conv5_cam.sum(1, keepdim=True).view(conv5.size(0), 1, mask_size, mask_size)
    mask_sum = F.interpolate(mask_sum, size=(img_size, img_size), mode='bilinear', align_corners=True)
    mask_sum = mask_sum.view(mask_sum.size(0), -1)
    x_range = mask_sum.max(-1, keepdim=True)[0] - mask_sum.min(-1, keepdim=True)[0]
    mask_sum = (mask_sum - mask_sum.min(-1, keepdim=True)[0]) / x_range
    mask_sum = mask_sum.view(mask_sum.size(0), 1, img_size, img_size)
    if mask is None:
        mask = 1 - torch.sign(torch.sign(mask_sum - rate) + 1)
    else:
        mask = mask * (1 - torch.sign(torch.sign(mask_sum - rate) + 1))
    return mask


def get_bbox(x, conv5, layer_weights, rate=0.3, img_size=448):
    mask_size = img_size // 32
    # conv5_cam = conv5.clone().detach().view(conv5.size(0), conv5.size(1), 14*14) * layer_weights.unsqueeze(-1)
    conv5_cam = conv5.clone().detach().view(conv5.size(0), conv5.size(1), -1) * layer_weights.unsqueeze(-1)

    mask_sum = conv5_cam.sum(1, keepdim=True).view(conv5.size(0), 1, mask_size, mask_size)
    mask_sum = F.interpolate(mask_sum, size=(img_size, img_size), mode='bilinear', align_corners=True)
    mask_sum = mask_sum.view(mask_sum.size(0), -1)
    x_range = mask_sum.max(-1, keepdim=True)[0] - mask_sum.min(-1, keepdim=True)[0]
    mask_sum = (mask_sum - mask_sum.min(-1, keepdim=True)[0]) / x_range
    mask = torch.sign(torch.sign(mask_sum - rate) + 1)
    mask = mask.view(mask.size(0), 1, img_size, img_size)
    input_box = torch.zeros_like(x)
    xy_list = []
    for k in torch.arange(x.size(0)):
        indices = mask[k].nonzero()
        y1, x1 = indices.min(dim=0)[0][-2:]
        y2, x2 = indices.max(dim=0)[0][-2:]
        tmp = x[k, :, y1:y2, x1:x2]
        if x1 == x2 or y1 == y2:
            tmp = x[k, :, :, :]
        input_box[k] = F.interpolate(tmp.unsqueeze(0), size=(img_size, img_size),
                                     mode='bilinear', align_corners=True).clone().detach().to(x.device)
        xy_list.append([x1, x2, y1, y2])
    return input_box, xy_list


class LocalCamNet(nn.Module):
    def __init__(self, config=None):
        super(LocalCamNet, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.box_thred = config.box_thred
        self.image_size = config.image_size

        # ----main branch----
        # basenet = getattr(import_module('torchvision.models'), config.arch)
        # basenet = basenet(pretrained=True)
        basenet = resnet50(pretrained=True)

        self.conv4 = nn.Sequential(*list(basenet.children())[:-3])
        self.conv5 = nn.Sequential(*list(basenet.children())[-3])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(2048, self.num_classes, bias=True)

        # ----other branch----
        self.conv4_box = copy.deepcopy(self.conv4)
        self.conv5_box = copy.deepcopy(self.conv5)
        self.classifier_box = Classifier(2048, self.num_classes, bias=True)

        self.conv4_box_2 = copy.deepcopy(self.conv4)
        self.conv5_box_2 = copy.deepcopy(self.conv5)

        self.classifier_box_2 = Classifier(2048, self.num_classes, bias=True)

        # part information
        self.conv6_1 = nn.Conv2d(1024, 10 * self.num_classes, 1, 1, 1)
        self.conv6_2 = nn.Conv2d(1024, 10 * self.num_classes, 1, 1, 1)
        self.conv6 = nn.Conv2d(1024, 10 * self.num_classes, 1, 1, 1)

        self.cls_part_1 = Classifier(10 * self.num_classes, self.num_classes, bias=True)
        self.cls_part_2 = Classifier(10 * self.num_classes, self.num_classes, bias=True)
        self.cls_part = Classifier(10 * self.num_classes, self.num_classes, bias=True)

        self.cls_cat_1 = Classifier(2048 + 10 * self.num_classes, self.num_classes, bias=True)
        self.cls_cat_2 = Classifier(2048 + 10 * self.num_classes, self.num_classes, bias=True)
        self.cls_cat = Classifier(2048 + 10 * self.num_classes, self.num_classes, bias=True)

        self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.cls_cat_a = Classifier(3 * (2048 + 10 * self.num_classes), self.num_classes, bias=True)

        # gating network
        self.conv4_gate = copy.deepcopy(self.conv4)
        self.conv5_gate = copy.deepcopy(self.conv5)
        self.cls_gate = nn.Sequential(Classifier(2048, 512, bias=True), Classifier(512, 3, bias=True))

    def forward(self, x, y=None, is_vis=False, vis_idx=None, gt_top=None):
        b = x.size(0)
        # ------main branch-----
        conv4 = self.conv4(x)
        conv5 = self.conv5(conv4)
        conv5_pool = self.pool(conv5).view(b, -1)
        logits = self.classifier(conv5_pool)

        # ---
        pool_conv6 = self.pool_max(F.relu(self.conv6(conv4.detach()))).view(b, -1)
        pool_cat = torch.cat((10 * l2_norm_v2(conv5_pool.detach()), 10 * l2_norm_v2(pool_conv6.detach())), dim=1)

        logits_max = self.cls_part(pool_conv6)
        logits_cat = self.cls_cat(pool_cat)

        # logits_cross = pool_conv6.view(b, self.num_classes, -1).mean(-1)

        # ------discriminative patch------
        # with torch.set_grad_enabled(True):
        with torch.enable_grad():
            layer_weights = None
            self.grad_cam = GradCam(model=self,
                                    feature_extractor=self.conv5,
                                    classifier=self.classifier,
                                    target_layer_names=["2"])
            target_index = y
            if is_vis and (not vis_idx is None):
                if vis_idx == 1:
                    target_index = gt_top
            layer_weights = self.grad_cam(conv4.detach(), target_index).to(x.device)
            self.grad_cam = None

        # ---bbox---
        input_box, box_xy = get_bbox(x, conv5, layer_weights, rate=self.box_thred, img_size=self.image_size)
        conv4_box = self.conv4_box(input_box.detach())
        conv5_box = self.conv5_box(conv4_box)
        conv5_box_pool = self.pool(conv5_box).view(b, -1)
        logits_box = self.classifier_box(conv5_box_pool)

        # ---
        pool_conv6_1 = self.pool_max(F.relu(self.conv6_1(conv4_box.detach()))).view(b, -1)
        pool_cat_1 = torch.cat((10 * l2_norm_v2(conv5_box_pool.detach()), 10 * l2_norm_v2(pool_conv6_1.detach())),
                               dim=1)

        logits_max_1 = self.cls_part_1(pool_conv6_1)
        logits_cat_1 = self.cls_cat_1(pool_cat_1)

        # logits_cross_1 = pool_conv6_1.view(b, self.num_classes, -1).mean(-1)

        # ------box 2------
        with torch.enable_grad():
            layer_weights_2 = None
            self.grad_cam = GradCam(model=self,
                                    feature_extractor=self.conv5_box,
                                    classifier=self.classifier_box,
                                    target_layer_names=["2"])
            target_index = y
            if is_vis and (not vis_idx is None):
                if vis_idx == 2:
                    target_index = gt_top
            layer_weights_2 = self.grad_cam(conv4_box.detach(), target_index)
            self.grad_cam = None

        input_box_2, box_xy_2 = get_bbox(
            input_box, conv5_box, layer_weights_2, rate=self.box_thred, img_size=self.image_size)
        conv4_box_2 = self.conv4_box_2(input_box_2.detach())
        conv5_box_2 = self.conv5_box_2(conv4_box_2)
        conv5_box_pool_2 = self.pool(conv5_box_2).view(b, -1)
        logits_box_2 = self.classifier_box_2(conv5_box_pool_2)

        # ---
        pool_conv6_2 = self.pool_max(F.relu(self.conv6_2(conv4_box_2.detach()))).view(b, -1)
        pool_cat_2 = torch.cat((10 * l2_norm_v2(conv5_box_pool_2.detach()), 10 * l2_norm_v2(pool_conv6_2.detach())),
                               dim=1)

        logits_max_2 = self.cls_part_2(pool_conv6_2)
        logits_cat_2 = self.cls_cat_2(pool_cat_2)

        # logits_cross_2 = pool_conv6_2.view(b, self.num_classes, -1).mean(-1)

        # gating network -------------
        conv5_gate = self.conv5_gate(self.conv4_gate(x))
        pool_gate = self.pool(conv5_gate).view(b, -1)
        pr_gate = F.softmax(self.cls_gate(pool_gate), dim=1)

        logits_gate = torch.stack([logits_cat.detach(), logits_cat_1.detach(), logits_cat_2.detach()], dim=-1)
        logits_gate = logits_gate * pr_gate.view(pr_gate.size(0), 1, pr_gate.size(1))
        logits_gate = logits_gate.sum(-1)

        # ------output------
        logits_list = [logits, logits_max, logits_cat,
                       logits_box, logits_max_1, logits_cat_1,
                       logits_box_2, logits_max_2, logits_cat_2,
                       logits_gate]

        outputs = {'logits': logits_list, 'pr_gate': pr_gate}

        return outputs

    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters()) + \
                           list(self.conv4.parameters()) + \
                           list(self.conv5_box.parameters()) + \
                           list(self.conv4_box.parameters()) + \
                           list(self.conv5_box_2.parameters()) + \
                           list(self.conv4_box_2.parameters()) + \
                           list(self.conv5_gate.parameters()) + \
                           list(self.conv4_gate.parameters())
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params


@MODEL.register
def MGE_CNN(config):
    return LocalCamNet(config)
