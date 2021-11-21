import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ScaleInvariant_Loss(predicted, groundtruth):
    """
    Referred to as 'scale-invariant loss' in the paper 'learning monocular dense depth from events' (3DV 2020)
    See also 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos'

    :param predicted:
    :param groundtruth:
    :return:
    """
    mask = ~torch.isnan(groundtruth)
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res[mask == False] = 0  # invalid pixels: nan --> 0

    MSE = 1 / n * torch.sum(torch.pow(res[mask], 2))
    quad = 1 / (n ** 2) * torch.pow(torch.sum(res[mask]), 2)

    return MSE - quad


def Multiscale_ScaleInvariant_Loss(predicted, groundtruth, factors=(1., 1., 1., 1.)):
    """

    :param predicted: a tuple of num_scales [N, 1, H, W] tensors
    :param groundtruth: a tuple of num_scales [N, 1, H, W] tensors
    :return: a scalar value
    """
    multiscale_loss = 0.0

    for (factor, map) in zip(factors, predicted):
        scale = (map.shape[-2], map.shape[-1])
        rescaled_gt = F.interpolate(groundtruth, size=scale, mode='bilinear', align_corners=False)
        multiscale_loss += factor * ScaleInvariant_Loss(map, rescaled_gt)

    return multiscale_loss


def GradientMatching_Loss(predicted, groundtruth):
    """
    Referred to as 'multi-scale scale-invariant gradient matching loss' in the paper 'learning monocular dense depth
    from events' (3DV 2020)
    See also 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos'

    :param predicted:
    :param groundtruth:
    :return:
    """
    mask = ~torch.isnan(groundtruth)
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res[mask == False] = 0  # invalid pixels: nan --> 0

    # define sobel filters for each direction
    if torch.cuda.is_available():
        sobelX = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).cuda()
        sobelY = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).cuda()
    else:
        sobelX = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3))
        sobelY = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3))

    # stride and padding of 1 to keep the same resolution
    grad_res_x = F.conv2d(res, sobelX, stride=1, padding=1)
    grad_res_y = F.conv2d(res, sobelY, stride=1, padding=1)

    # use the value of the gradients only at valid pixel locations
    grad_res_x *= mask
    grad_res_y *= mask

    return 1 / n * torch.sum(torch.abs(grad_res_x[mask]) + torch.abs(grad_res_y[mask]))


def MultiScale_GradientMatching_Loss(predicted, groundtruth, factors=(1., 1., 1., 1.)):
    """
    Computes the gradient matching loss at each scale, then return the sum.

    :param predicted: a tuple of num_scales [N, 1, H, W] tensors
    :param groundtruth: a tuple of num_scales [N, 1, H, W] tensors
    :return: a scalar value
    """
    multiscale_loss = 0.0

    for (factor, map) in zip(factors, predicted):
        scale = (map.shape[-2], map.shape[-1])
        rescaled_gt = F.interpolate(groundtruth, size=scale, mode='bilinear', align_corners=False)
        multiscale_loss += factor * GradientMatching_Loss(map, rescaled_gt)

    return multiscale_loss


def SpikePenalization_Loss(intermediary_spike_tensors):
    """
    Regularization loss to diminish the spiking activity of the network. Penalizes the square of the mean spike counts.

    :param intermediary_spike_tensors: a list of integer spike tensors
    """
    spk_penalization_loss = 0.0

    for spike_tensor in intermediary_spike_tensors:
        spk_penalization_loss += 1/(2 * spike_tensor.numel()) * torch.sum(torch.pow(spike_tensor, 2))

    return spk_penalization_loss


class Total_Loss(nn.Module):
    """
    For learning linear (metric) depth, use alpha=0.5
    Tests were done without any weighting of predictions at different scales --> scale_weights = (1., 1., 1., 1.)

    Spike penalization can be balanced with beta weight parameter. Increasing it will reduce spiking activity and
    accuracy.
    """

    def __init__(self, alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=False, beta=1.):
        super().__init__()
        self.alpha = alpha
        self.scale_weights = scale_weights
        self.penalize_spikes = penalize_spikes
        self.beta = beta

    def forward(self, predicted, groundtruth, intermediary_spike_tensors=None):

        if not self.penalize_spikes:
            return Multiscale_ScaleInvariant_Loss(predicted, groundtruth, self.scale_weights) + \
                   self.alpha * MultiScale_GradientMatching_Loss(predicted, groundtruth, self.scale_weights)

        else:
            return Multiscale_ScaleInvariant_Loss(predicted, groundtruth, self.scale_weights) + \
                self.alpha * MultiScale_GradientMatching_Loss(predicted, groundtruth, self.scale_weights) + \
                self.beta * SpikePenalization_Loss(intermediary_spike_tensors)
