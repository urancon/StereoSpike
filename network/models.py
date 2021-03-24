import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate  # , accelerating
from spikingjelly.clock_driven.neuron import BaseNode, IFNode, LIFNode
# from test_blocks import PLIFNode
from torchvision import transforms
import math


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


class MonoSpikeFlowNetLike(nn.Module):
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

    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=1000):
        super().__init__()
        # SNN-related parameters
        self.T = T

        # encoder layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(256),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(512),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        # residual layers
        self.conv_r11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(512),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.conv_r12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(512),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.conv_r21 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(512),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.conv_r22 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(512),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        # decoder layers
        # (upsampling layers)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256+128+1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128+64+1, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        infinite_thresh = 1000
        self.predict_depth4 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.predict_depth3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.predict_depth2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.predict_depth1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        self.upsampled_depth4_to_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.upsampled_depth3_to_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384+1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.upsampled_depth2_to_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192+1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.upsampled_depth1_to_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=68+1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )


    def forward(self, x):
        # TODO: in Spike-FlowNet, "the loss is evaluated after forward-propagating all consecutive input event frames
        #   within the time window"

        # TODO: for our depth prediction task on MVSEC, membrane potentials should NOT be reset

        # TODO: use spikingjelly's layer-by-layer mode, instead of step-by-step as it is now

        # TODO: on my personal computer, too much memory is needed for a simple inference on 1 frame with T=100 !!! we
        #  need to know the memory limitations of our data representation

        for t in range(self.T):

            # pass through encoder layers
            out_conv1 = self.conv1(x[t])
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

            # pass through residual blocks
            out_rconv11 = self.conv_r11(out_conv4)
            out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
            out_rconv21 = self.conv_r21(out_rconv12)
            out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

            # gradually upsample while concatenating and passing through skip connections
            depth4 = self.predict_depth4(self.upsampled_depth4_to_3(out_rconv22))
            depth4_up = self.crop_like(depth4, out_conv3)
            out_deconv3 = self.crop_like(self.deconv3(out_rconv22), out_conv3)
            concat3 = torch.cat((out_conv3, out_deconv3, depth4_up), 1)

            depth3 = self.predict_depth3(self.upsampled_depth3_to_2(concat3))
            depth3_up = self.crop_like(depth3, out_conv2)
            out_deconv2 = self.crop_like(self.deconv2(concat3), out_conv2)
            concat2 = torch.cat((out_conv2, out_deconv2, depth3_up), 1)

            depth2 = self.predict_depth2(self.upsampled_depth2_to_1(concat2))
            depth2_up = self.crop_like(depth2, out_conv1)
            out_deconv1 = self.crop_like(self.deconv1(concat2), out_conv1)
            concat1 = torch.cat((out_conv1, out_deconv1, depth2_up), 1)

            depth1 = self.predict_depth1(self.upsampled_depth1_to_0(concat1))

        return depth1


if __name__ == "__main__":
    # from mvsec_utils.mvsec_dataset import MVSEC

    #device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'

    T = 50

    net = MonoSpikeFlowNetLike(tau=10, T=T, v_threshold=1.0, v_reset=0.0, v_infinite_thresh=1000).to(device)

    frame = torch.rand(T, 1, 2, 360, 246).to(device)  # input tensor of shape [T, N_batches, 2 (polarities), W, H]
    output = net(frame)

    print("output.shape:", output.shape)  # output is a single-channel depth map of size [W, H]

    assert output.shape[2:] == frame.shape[3:], \
        "input and output do not have the same shape: respectively {} and {}".format(frame.shape[3:], output.shape[2:])

    print("ok")

