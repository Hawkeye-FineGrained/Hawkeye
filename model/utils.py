import torch
import torch.nn as nn


def initialize_weights(m) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, val=0)


def initialize_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))


def load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def freeze(m):
    if isinstance(m, nn.Module):
        for param in m.parameters():
            param.require_grad = False
    elif isinstance(m, list):
        for param in m:
            param.require_grad = False
    else:
        raise NotImplementedError()


def unfreeze(m):
    if isinstance(m, nn.Module):
        for param in m.parameters():
            param.require_grad = True
    elif isinstance(m, list):
        for param in m:
            param.require_grad = True
    else:
        raise NotImplementedError()