import torch
import random
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from model.backbone import resnet50
from model.registry import MODEL


def makeGaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


class KernelGenerator(nn.Module):
    def __init__(self, size, offset=None):
        super(KernelGenerator, self).__init__()

        self.size = self._pair(size)
        xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
        if offset is None:
            offset_x = offset_y = size // 2
        else:
            offset_x, offset_y = self._pair(offset)
        self.factor = torch.from_numpy(-(np.power(xx - offset_x, 2) + np.power(yy - offset_y, 2)) / 2).float()

    @staticmethod
    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    def forward(self, theta):
        pow2 = torch.pow(theta * self.size[0], 2)
        kernel = 1.0 / (2 * np.pi * pow2) * torch.exp(self.factor.to(theta.device) / pow2)
        return kernel / kernel.max()


def kernel_generate(theta, size, offset=None):
    return KernelGenerator(size, offset)(theta)


def _mean_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices = F.max_pool2d(
            padded_maps,
            kernel_size=win_size,
            stride=1,
            return_indices=True,
            ceil_mode=True)
        peak_map = (indices == element_map)

        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)

        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                   peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1) / \
                     (peak_map.view(batch_size, num_channels, -1).sum(2).view(batch_size, num_channels, 1, 1) + 1e-6)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)


class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@MODEL.register
class S3N(nn.Module):

    def __init__(self, config):
        super(S3N, self).__init__()
        self.config = config
        num_classes = self.config.num_classes

        self.backbone = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.num_features = 2048

        self.grid_size = 31
        self.padding_size = 30

        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size_net = self.config.image_size
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm=13))
        self.base_ratio = self.config.base_ratio
        self.radius = ScaleLayer(self.config.radius)
        self.radius_inv = ScaleLayer(self.config.radius_inv)

        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:, :, :] = gaussian_weights

        self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_size - 1.0) + (1.0 - k) * (
                            j - self.padding_size) / (self.grid_size - 1.0)

        self.raw_classifier = nn.Linear(2048, num_classes)
        self.sampler_buffer = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
        self.sampler_classifier = nn.Linear(2048, num_classes)

        self.sampler_buffer1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
        self.sampler_classifier1 = nn.Linear(2048, num_classes)

        self.con_classifier = nn.Linear(int(self.num_features * 3), num_classes)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.map_origin = nn.Conv2d(2048, num_classes, 1, 1, 0)

    def create_grid(self, x):
        P = torch.autograd.Variable(
            torch.zeros(1, 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size).to(
                x.device), requires_grad=False)
        P[0, :, :, :] = self.P_basis
        P = P.expand(x.size(0), 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)

        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)

        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)

        x_filter = x_filter / p_filter
        y_filter = y_filter / p_filter

        xgrids = x_filter * 2 - 1
        ygrids = y_filter * 2 - 1
        xgrids = torch.clamp(xgrids, min=-1, max=1)
        ygrids = torch.clamp(ygrids, min=-1, max=1)

        xgrids = xgrids.view(-1, 1, self.grid_size, self.grid_size)
        ygrids = ygrids.view(-1, 1, self.grid_size, self.grid_size)

        grid = torch.cat((xgrids, ygrids), 1)

        grid = F.interpolate(grid, size=(self.input_size_net, self.input_size_net), mode='bilinear', align_corners=True)

        grid = torch.transpose(grid, 1, 2)
        grid = torch.transpose(grid, 2, 3)

        return grid

    def generate_map(self, input_x, class_response_maps, p):
        N, C, H, W = class_response_maps.size()
        device = input_x.device

        score_pred, sort_number = torch.sort(F.softmax(F.adaptive_avg_pool2d(class_response_maps, 1), dim=1), dim=1,
                                             descending=True)
        gate_score = (score_pred[:, 0:5] * torch.log(score_pred[:, 0:5])).sum(1)

        xs = []
        xs_inv = []

        for idx_i in range(N):
            if gate_score[idx_i] > -0.2:
                decide_map = class_response_maps[idx_i, sort_number[idx_i, 0], :, :]
            else:
                decide_map = class_response_maps[idx_i, sort_number[idx_i, 0:5], :, :].mean(0)

            min_value, max_value = decide_map.min(), decide_map.max()
            decide_map = (decide_map - min_value) / (max_value - min_value)

            peak_list, aggregation = peak_stimulation(decide_map, win_size=3, peak_filter=_mean_filter)

            decide_map = decide_map.squeeze(0).squeeze(0)

            score = [decide_map[item[2], item[3]] for item in peak_list]
            x = [item[3] for item in peak_list]
            y = [item[2] for item in peak_list]

            if score == []:
                temp = torch.zeros(1, 1, self.grid_size, self.grid_size).to(device)
                temp += self.base_ratio
                xs.append(temp)
                continue

            peak_num = torch.arange(len(score))

            temp = self.base_ratio
            temp_w = self.base_ratio

            if p == 0:
                for i in peak_num:
                    temp += score[i] * kernel_generate(self.radius(torch.sqrt(score[i])), H,
                                                       (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).to(device)
                    temp_w += 1 / score[i] * \
                              kernel_generate(self.radius_inv(torch.sqrt(score[i])), H,
                                              (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).to(device)
            elif p == 1:
                for i in peak_num:
                    rd = random.uniform(0, 1)
                    if score[i] > rd:
                        temp += score[i] * kernel_generate(self.radius(torch.sqrt(score[i])), H,
                                                           (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).to(
                            device)
                    else:
                        temp_w += 1 / score[i] * \
                                  kernel_generate(self.radius_inv(torch.sqrt(score[i])), H,
                                                  (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).to(device)
            elif p == 2:
                index = score.index(max(score))
                temp += score[index] * kernel_generate(self.radius(torch.sqrt(score[index])), H,
                                                       (x[index].item(), y[index].item())).unsqueeze(0).unsqueeze(0).to(
                    device)

                index = score.index(min(score))
                temp_w += 1 / score[index] * \
                          kernel_generate(self.radius_inv(torch.sqrt(score[index])), H,
                                          (x[index].item(), y[index].item())).unsqueeze(0).unsqueeze(0).to(device)

            if type(temp) == float:
                temp += torch.zeros(1, 1, self.grid_size, self.grid_size).to(device)
            xs.append(temp)

            if type(temp_w) == float:
                temp_w += torch.zeros(1, 1, self.grid_size, self.grid_size).to(device)
            xs_inv.append(temp_w)

        xs = torch.cat(xs, 0)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid = self.create_grid(xs_hm).to(device)
        x_sampled_zoom = F.grid_sample(input_x, grid, align_corners=True)

        xs_inv = torch.cat(xs_inv, 0)
        xs_hm_inv = nn.ReplicationPad2d(self.padding_size)(xs_inv)
        grid_inv = self.create_grid(xs_hm_inv).to(device)
        x_sampled_inv = F.grid_sample(input_x, grid_inv, align_corners=True)

        return x_sampled_zoom, x_sampled_inv

    def forward(self, input_x, p):

        self.map_origin.weight.data.copy_(self.raw_classifier.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.map_origin.bias.data.copy_(self.raw_classifier.bias.data)

        feature_raw = self.features(input_x)
        agg_origin = self.raw_classifier(self.avg(feature_raw).view(-1, 2048))

        with torch.no_grad():
            class_response_maps = F.interpolate(self.map_origin(feature_raw), size=self.grid_size, mode='bilinear',
                                                align_corners=True)
        x_sampled_zoom, x_sampled_inv = self.generate_map(input_x, class_response_maps, p)

        feature_D = self.sampler_buffer(self.features(x_sampled_zoom))
        agg_sampler = self.sampler_classifier(self.avg(feature_D).view(-1, 2048))

        feature_C = self.sampler_buffer1(self.features(x_sampled_inv))
        agg_sampler1 = self.sampler_classifier1(self.avg(feature_C).view(-1, 2048))

        aggregation = self.con_classifier(torch.cat(
            [self.avg(feature_raw).view(-1, 2048), self.avg(feature_D).view(-1, 2048),
             self.avg(feature_C).view(-1, 2048)], 1))

        return aggregation, agg_origin, agg_sampler, agg_sampler1
