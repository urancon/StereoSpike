import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from spikingjelly.clock_driven import functional, layer, surrogate  # , accelerating
from spikingjelly.clock_driven.neuron import BaseNode, IFNode, LIFNode
# from blocks import PLIFNode


class NeuromorphicNet(nn.Module):

    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):  # TODO: change forward function according to our data format and to our model
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        out_spikes_counter = self.boost(self.fc(self.conv(x[0])).unsqueeze(1)).squeeze(1)
        for t in range(1, x.shape[0]):
            out_spikes_counter += self.boost(self.fc(self.conv(x[t])).unsqueeze(1)).squeeze(1)
        return out_spikes_counter


class DepthSNN(NeuromorphicNet):
    def __init__(self):
        pass


class SpikeFlowNetLike(nn.Module):
    """
    A full spiking Spike-FlowNet network, but for monocular depth estimation.
    Its architecture is very much like a U-Net, and it outputs a single-channel depth map of the same size as the input
    field of view.
    Basically, we have replaced all ReLU activation functions in Spike-FlowNet by LIFNodes -> non-linearity
    Intermediate depth maps (see paper) come out of IFNodes with an infinite threshold.
    """

    @staticmethod
    def crop_like(input_tensor, target):
        if input_tensor.size()[2:] == target.size()[2:]:
            return input_tensor
        else:
            return input_tensor[:, :, :target.size(2), :target.size(3)]

    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf')):
        super().__init__()
        # SNN-related parameters
        self.T = T

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(512),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD'),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            OneToOne((260, 346)),
            neuron.IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                m.v.detach_()
            elif isinstance(m, layer.Dropout):
                m.mask.detach_()

    def forward(self, x):
        #  TODO: in Spike-FlowNet, "the loss is evaluated after forward-propagating all consecutive input event frames
        #   within the time window"

        # TODO: use spikingjelly's layer-by-layer mode, instead of step-by-step as it is now

        # TODO: on my personal computer, too much memory is needed for a simple inference on 1 frame with T=100 !!! we
        #  need to know the memory limitations of our data representation

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # pass through encoder layers
            out_conv1 = self.conv1(frame)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

            # pass through residual blocks
            out_rconv = self.bottleneck(out_conv4)

            # gradually upsample while concatenating and passing through skip connections
            out_deconv4 = self.deconv4(out_rconv)
            out_add4 = self.crop_like(out_deconv4, out_conv3) + out_conv3

            out_deconv3 = self.deconv3(out_add4)
            out_add3 = self.crop_like(out_deconv3, out_conv2) + out_conv2

            out_deconv2 = self.deconv2(out_add3)
            out_add2 = self.crop_like(out_deconv2, out_conv1) + out_conv1

            out_deconv1 = self.deconv1(out_add2)

            out_depth1 = self.predict_depth(out_deconv1)

        # the membrane potentials of the output IF neuron carry the depth prediction
        return self.predict_depth[-1].v


class FusionFlowNetLike(nn.Module):
    """
    A fully spiking Fusion-FlowNet network, but for binocular depth estimation.
    Its architecture is very much like a U-Net, and it outputs a single-channel depth map of the same size as the input
    field of view. The main difference with a U-Net is in the encoder, which takes as inputs 2 DVS spike trains.
    Basically, we have replaced all ReLU activation functions in Fusion-FlowNet by LIFNodes -> non-linearity.
    The stream of spikes of the second DVS camera replaces the grayscale image that is fed to the network in
    Fusion-FlowNet. Concatenations are also replaced by additions.
    Intermediate depth maps (see paper) come out of IFNodes with an infinite threshold.
    """

    @staticmethod
    def crop_like(input_tensor, target):
        if input_tensor.size()[2:] == target.size()[2:]:
            return input_tensor
        else:
            return input_tensor[:, :, :target.size(2), :target.size(3)]

    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf')):
        super().__init__()
        # SNN-related parameters
        self.T = T

        # encoder layers (downsampling) for the left input spiketrain
        self.conv1_left = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(32),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2_left = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3_left = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4_left = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling) for the right input spiketrain
        self.conv1_right = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(32),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2_right = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3_right = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4_right = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD'),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            neuron.IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                m.v.detach_()
            elif isinstance(m, layer.Dropout):
                m.mask.detach_()

    def forward(self, x_left, x_right):
        # x_left and x_right must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x_left.shape[1]):

            # separate left and right input spiketrains
            left_frame = x_left[:, t, :, :, :][0]
            right_frame = x_right[:, t, :, :, :][0]

            # pass through left encoder layers
            out_conv1_left = self.conv1_left(left_frame)
            out_conv2_left = self.conv2_left(out_conv1_left)
            out_conv3_left = self.conv3_left(out_conv2_left)
            out_conv4_left = self.conv4_left(out_conv3_left)

            # pass through right encoder layers
            out_conv1_right = self.conv1_right(right_frame)
            out_conv2_right = self.conv2_right(out_conv1_right)
            out_conv3_right = self.conv3_right(out_conv2_right)
            out_conv4_right = self.conv4_right(out_conv3_right)

            # concatenate the features calculated by both pathways
            out_conv4 = torch.cat(out_conv4_left, out_conv4_right)

            # pass through residual blocks
            out_rconv = self.bottleneck(out_conv4)

            # gradually upsample while summing spikes and passing through skip connections
            out_deconv4 = self.deconv4(out_rconv)
            out_add4 = self.crop_like(out_deconv4, out_conv3_left) + out_conv3_left + out_conv3_right

            out_deconv3 = self.deconv3(out_add4)
            out_add3 = self.crop_like(out_deconv3, out_conv2_left) + out_conv2_left + out_conv2_right

            out_deconv2 = self.deconv2(out_add3)
            out_add2 = self.crop_like(out_deconv2, out_conv1_left) + out_conv1_left + out_conv1_right

            out_deconv1 = self.deconv1(out_add2)

            out_depth1 = self.predict_depth(out_deconv1)

        # the membrane potentials of the output IF neuron carry the depth prediction
        return self.predict_depth[-1].v

