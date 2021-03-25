import numpy as np
import cv2
import torch
import torch.nn.functional as F


def MeanDepthError(predicted, groundtruth):
    """
    The Mean Depth Error (MDE) is commonly used as a metric to evaluate depth estimation algorithms on MVSEC dataset.

    :param predicted: a single-channel torch array of shape [W, H]
    :param groundtruth: a a single-channel torch array of shape [W, H]
    :return: the MDE between the given prediction and label
    """
    return torch.mean(torch.abs(predicted - groundtruth))


def ScaleInvariant_Loss(predicted, groundtruth):
    """
    Referred to as 'scale-invariant loss' in the paper 'learning monocular dense depth from events' (3DV 2020)
    See also 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos'

    :param predicted:
    :param groundtruth:
    :return:
    """
    n = predicted.numel()
    res = predicted - groundtruth
    MSE = 1/n * torch.sum(torch.pow(res, 2))
    quad = 1/(n ** 2) * torch.pow(torch.sum(res), 2)
    return MSE - quad


def GradientMatching_Loss(predicted, groundtruth):
    """
    Referred to as 'multi-scale scale-invariant gradient matching loss' in the paper 'learning monocular dense depth
    from events' (3DV 2020)
    See also 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos'

    TODO: according to the preceding paper, this loss should actually be computed at different scales and be equal to
     the sum of its values at each scale

    :param predicted:
    :param groundtruth:
    :return:
    """
    n = predicted.numel()
    res = predicted - groundtruth
    grad_res_x = cv2.Sobel(res, 0, dx=1, dy=0)
    grad_res_y = cv2.Sobel(res, 0, dx=0, dy=1)
    return 1/n * torch.sum(torch.abs(grad_res_x) + torch.abs(grad_res_y))


def Total_Loss(predicted, groundtruth, alpha=0.5):
    return ScaleInvariant_Loss(predicted, groundtruth) + alpha * GradientMatching_Loss(predicted, groundtruth)

