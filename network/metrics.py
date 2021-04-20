import numpy as np
import cv2
import torch
import torch.nn.functional as F


def mask_dead_pixels(predicted, groundtruth):
    """
    Some pixels of the groundtruth have values of zero or NaN, which has no physical meaning. This function masks them.
    All such invalid values become 0, and valid ones remain as they are.

    :param predicted:
    :param groundtruth:
    :return:
    """
    assert predicted.shape == groundtruth.shape, "input and target tensors do not have the same shape, can't apply " \
                                                 "the same mask to them ! " \
                                                 "Input is of shape {} and target of shape {}".format(predicted.shape, groundtruth.shape)
    mask = (groundtruth != 0) & (groundtruth != 255)  # only consider valid groundtruth pixels
    predicted *= mask
    groundtruth *= mask
    return predicted, groundtruth


def lin_to_log_depths(depths_rect_lin, Dmax=10, alpha=9.2):
    """
    Applies normalized logarithm to the input depth maps. Log depth maps can represent large variations in a compact
    range, hence facilitating learning.
    Refer to the paper 'Learning Monocular Depth from Events' (3DV 2020) for more details.

    Basically,
    Dlin = Dmax * exp(-alpha * (1-Dlog))

    We only do this operation on elements that are different from 0 and 255, since those values (although present in the
     dataset) have no physical meaning and hinder learning

    With Dmax=10 and alpha=9.2, the minimum depth that can be predicted is of 0.001 meter.

    Also, predicted log depth should belong to [0; 1] interval.

    :param depths_rect_lin:  a tensor of shape [# of depth maps, W, H] containing depth maps at original linear scale
    :param Dmax: maximum expected depth
    :param alpha: parameter such that depth value of 0 maps to minimum observed depth. The bigger the alpha, the better
    the depth resolution.
    :return:  a tensor of shape [# of depth maps, W, H], but containing normalized log values instead
    """
    mask = (depths_rect_lin != 0) & (depths_rect_lin != 255)
    depths_rect_lin[mask] = 1 - 1 / alpha * (np.log(Dmax) - np.log(depths_rect_lin[mask]))
    return depths_rect_lin


def log_to_lin_depths(depths_rect_log, Dmax=10, alpha=9.2):
    """
    Inverse operation. Takes log depth maps, return lin depth maps, while ignoring invalid pixels

    :param depths_rect_log:
    :param Dmax:
    :param alpha:
    :return:
    """
    mask = (depths_rect_log != 0) & (depths_rect_log != 255)
    depths_rect_log[mask] = Dmax * np.exp(-alpha * (1 - depths_rect_log[mask]))
    return depths_rect_log


def MeanDepthError(predicted, groundtruth):
    """
    The Mean Depth Error (MDE) is commonly used as a metric to evaluate depth estimation algorithms on MVSEC dataset.
    It is computed only on non-NaN and non-zero values of the groundtruth depth map

    TODO: for the moment, assumes that the groundtruth depth and predictions are linear. If they are logarithmic, you
     will have to convert the result to linear scale in order to get a metric whose unit is meter.

    :param predicted: a single-channel torch array of shape [W, H]
    :param groundtruth: a a single-channel torch array of shape [W, H]
    :return: the MDE between the given prediction and label
    """
    if type(predicted) == tuple:
        predicted = predicted[0]
    mask = (groundtruth != 0) & (groundtruth != 255)  # only consider valid groundtruth pixels
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res = res * mask
    return torch.sum(torch.abs(res))/n


def ScaleInvariant_Loss(predicted, groundtruth):
    """
    Referred to as 'scale-invariant loss' in the paper 'learning monocular dense depth from events' (3DV 2020)
    See also 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos'

    :param predicted:
    :param groundtruth:
    :return:
    """
    mask = (groundtruth != 0) & (groundtruth != 255)  # only consider valid groundtruth pixels
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res = res * mask

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
    mask = (groundtruth != 0) & (groundtruth != 255)  # only consider valid groundtruth pixels
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res = res * mask
    res = res.view((1, 1, groundtruth.shape[-2], groundtruth.shape[-1]))  # (1, 1, H, W)

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

    return 1/n * torch.sum(torch.abs(grad_res_x) + torch.abs(grad_res_y))


def MultiScale_GradientMatching_Loss(predicted, groundtruth):
    """
    Computes the gradient matching loss at each scale, then return the sum.

    :param predicted:
    :param groundtruth:
    :return:
    """
    multiscale_loss = 0.0

    for map in predicted:
        scale = (map.shape[-2], map.shape[-1])
        rescaled_gt = F.interpolate(groundtruth.unsqueeze(0), size=scale, mode='bilinear', align_corners=False)
        multiscale_loss += GradientMatching_Loss(map, rescaled_gt)

    return multiscale_loss


def Total_Loss(predicted, groundtruth, alpha=0.5):
    return ScaleInvariant_Loss(predicted[0], groundtruth) + alpha * MultiScale_GradientMatching_Loss(predicted, groundtruth)

