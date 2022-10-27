import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy import stats

# global variable for speeding up the calculation
prev_bs = 0
prior_dist = None


class InterpPartsLoss(nn.Module):
    def __init__(self, config):
        super(InterpPartsLoss, self).__init__()
        self.radius = config.radius if 'radius' in config else 2  # Radius for the Gaussian kernel.
        self.std = config.std if 'std' in config else 0.4  # Standard deviation for the Gaussian kernel.
        self.num_parts = config.num_parts if 'num_parts' in config else 5  # Number of parts.
        self.alpha = config.alpha if 'alpha' in config else 1  # Parameter of Beta distribution.
        self.beta = config.beta if 'beta' in config else 0.001  # Parameter of Beta distribution.
        self.coeff = config.coeff if 'coeff' in config else 0.5  # Weight of loss.

        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits, att, assign = output

        loss_ce = self.ce_loss(logits, target)
        shaping_loss = ShapingLoss(assign, self.radius, self.std, self.num_parts, self.alpha, self.beta)

        loss = loss_ce + self.coeff * shaping_loss
        return loss


def GaussianKernel(radius, std):
    """
    Generate a gaussian blur kernel based on the given radius and std.

    Args
    ----------
    radius: int
        Radius of the Gaussian kernel. Center not included.
    std: float
        Standard deviation of the Gaussian kernel.

    Returns
    ----------
    weight: torch.FloatTensor, [2 * radius + 1, 2 * radius + 1]
        Output Gaussian kernel.

    """
    size = 2 * radius + 1
    weight = torch.ones(size, size)
    weight.requires_grad = False
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            dis = (i * i) + (j * j)
            weight[i + radius][j + radius] = np.exp(-dis / (2 * std * std))
    weight = weight / weight.sum()
    return weight


def update_prior_dist(batch_size, alpha, beta, device=None):
    """
    Update the samples of prior distribution due to the change of batchsize.

    Args
    ----------
    batch_size: int
        Current batch size.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.

    """
    global prior_dist
    grid_points = torch.arange(1., 2 * batch_size, 2.).float().cuda(device) / (2 * batch_size)
    grid_points_np = grid_points.cpu().numpy()
    grid_points_icdf = stats.beta.ppf(grid_points_np, a=alpha, b=beta)
    prior_dist = torch.tensor(grid_points_icdf).float().cuda(device).unsqueeze(1)


def ShapingLoss(assign, radius, std, num_parts, alpha, beta, eps=1e-5):
    """
    Wasserstein shaping loss for Bernoulli distribution.

    Args
    ----------
    assign: torch.cuda.FloatTensor, [batch_size, num_parts, height, width]
        Assignment map for grouping.
    radius: int
        Radius for the Gaussian kernel.
    std: float
        Standard deviation for the Gaussian kernel.
    num_parts: int
        Number of object parts in the current model.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.
    eps:
        Epsilon for rescaling the distribution.

    Returns
    ----------
    loss: torch.cuda.FloatTensor, [1, ]
        Average Wasserstein shaping loss for the current minibatch.

    """
    global prev_bs, prior_dist
    batch_size = assign.shape[0]
    device = assign.device

    # Gaussian blur
    if radius == 0:
        assign_smooth = assign
    else:
        weight = GaussianKernel(radius, std)
        weight = weight.contiguous().view(1, 1, 2 * radius + 1, 2 * radius + 1).expand(num_parts, 1, 2 * radius + 1,
                                                                                       2 * radius + 1).cuda(device)
        assign_smooth = F.conv2d(assign, weight, groups=num_parts)

    # pool the assignment maps into [batch_size, num_parts] for the empirical distribution of part occurence
    part_occ = F.adaptive_max_pool2d(assign_smooth, (1, 1)).squeeze(2).squeeze(2)
    emp_dist, _ = part_occ.sort(dim=0, descending=False)

    # the Beta prior
    if batch_size != prev_bs:
        update_prior_dist(batch_size, alpha, beta, device)

    # rescale the distribution
    emp_dist = (emp_dist + eps).log()
    prior_dist = (prior_dist + eps).log()

    # return the loss
    output_nk = (emp_dist - prior_dist).abs()
    loss = output_nk.mean()
    return loss
