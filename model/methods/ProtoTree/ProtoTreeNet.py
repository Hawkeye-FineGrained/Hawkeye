import copy
import torch
from torch import nn

from model.backbone import resnet50
from model.methods.ProtoTree.prototree import ProtoTree
from model.utils import initialize_weights_xavier
from model.registry import MODEL


@MODEL.register
class ProtoTreeNet(torch.nn.Module):

    def __init__(self, config):
        super(ProtoTreeNet, self).__init__()
        self.config = config

        resnet = resnet50(pretrained=True)
        state_dict = self.get_inat_resnet50_weight(config.backbone.pretrain)
        resnet.load_state_dict(state_dict, strict=False)
        neck_conv_in_channels = [i for i in resnet.modules() if isinstance(i, nn.Conv2d)][-1].out_channels

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.neck_conv = nn.Sequential(
            nn.Conv2d(in_channels=neck_conv_in_channels, out_channels=config.num_features, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.tree = ProtoTree(args=config)

        # init proto and neck_conv
        with torch.no_grad():
            torch.nn.init.normal_(self.tree.prototype_layer.prototype_vectors, mean=0.5, std=0.1)
            self.neck_conv.apply(initialize_weights_xavier)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck_conv(features)
        pred, info = self.tree(x, features)
        return pred, info

    def get_inat_resnet50_weight(self, pretrain):
        model_dict = torch.load(pretrain)
        # rename last residual block from cb_block to layer4.2
        new_model = copy.deepcopy(model_dict)
        for k in model_dict.keys():
            if k.startswith('module.backbone.cb_block'):
                splitted = k.split('cb_block')
                new_model['layer4.2' + splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith('module.backbone.rb_block'):
                del new_model[k]
            elif k.startswith('module.backbone.'):
                splitted = k.split('backbone.')
                new_model[splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith('module.classifier'):
                del new_model[k]
        # print(new_model.keys())
        return new_model

