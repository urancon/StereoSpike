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
    # mask = (groundtruth != 0) & (groundtruth != 255) & (groundtruth != 1000)  # only consider valid groundtruth pixels
    mask = ~torch.isnan(groundtruth)
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    # res = res * mask
    res[mask == False] = 0  # invalid pixels: nan --> 0

    MSE = 1 / n * torch.sum(torch.pow(res[mask], 2))
    quad = 1 / (n ** 2) * torch.pow(torch.sum(res[mask]), 2)

    return MSE - quad


def Multiscale_ScaleInvariant_Loss(predicted, groundtruth):
    """

    :param predicted: a tuple of num_scales [N, 1, H, W] tensors
    :param groundtruth: a tuple of num_scales [N, 1, H, W] tensors
    :return: a scalar value
    """
    multiscale_loss = 0.0

    for map in predicted:
        scale = (map.shape[-2], map.shape[-1])
        rescaled_gt = F.interpolate(groundtruth, size=scale, mode='bilinear', align_corners=False)
        multiscale_loss += ScaleInvariant_Loss(map, rescaled_gt)

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
    # mask = (groundtruth != 0) & (groundtruth != 255) & (groundtruth != 1000)  # only consider valid groundtruth pixels
    mask = ~torch.isnan(groundtruth)
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    # res = res * mask
    res[mask == False] = 0  # invalid pixels: nan --> 0

    # res = res.view((groundtruth.shape[-3], 1, groundtruth.shape[-2], groundtruth.shape[-1]))  # (N, 1, H, W)

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

    # come back to the original mask's shape : [batch_size, 1, H, W] --> [1, batch_size, H, W]
    # grad_res_x = grad_res_x.permute(1, 0, 2, 3)
    # grad_res_y = grad_res_y.permute(1, 0, 2, 3)

    # use the value of the gradients only at valid pixel locations
    grad_res_x *= mask
    grad_res_y *= mask

    return 1 / n * torch.sum(torch.abs(grad_res_x[mask]) + torch.abs(grad_res_y[mask]))


def MultiScale_GradientMatching_Loss(predicted, groundtruth):
    """
    Computes the gradient matching loss at each scale, then return the sum.

    :param predicted: a tuple of num_scales [N, 1, H, W] tensors
    :param groundtruth: a tuple of num_scales [N, 1, H, W] tensors
    :return: a scalar value
    """
    multiscale_loss = 0.0

    for map in predicted:
        scale = (map.shape[-2], map.shape[-1])
        rescaled_gt = F.interpolate(groundtruth, size=scale, mode='bilinear', align_corners=False)
        multiscale_loss += GradientMatching_Loss(map, rescaled_gt)

    return multiscale_loss


class Total_Loss(nn.Module):
    """
    For learning linear (metric) depth, use alpha=0.5
    """

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, predicted, groundtruth):
        return Multiscale_ScaleInvariant_Loss(predicted, groundtruth) + \
               self.alpha * MultiScale_GradientMatching_Loss(predicted, groundtruth)
