import torch
from torch import nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, neuron, layer, surrogate

from .blocks import ResBlock, NNConvUpsampling, BilinConvUpsampling


class AnalogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_test_accuracy = float('inf')
        self.epoch = 0

    def increment_epoch(self):
        self.epoch += 1

    def get_max_accuracy(self):
        return self.max_test_accuracy

    def update_max_accuracy(self, new_acc):
        self.max_test_accuracy = new_acc

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SteroSpike_equivalentANN(AnalogNet):
    """
    An Analog Neural Network (ANN) with the exact same architecture as StereoSpike.
    Uses biases in convolution layers, batch normalization, and classical activation functions such as Sigmoid.

    Such equivalent ANNs show worse performances than StereoSpike, which is a SNN.
    """

    def __init__(self, activation_function=nn.Sigmoid()):
        super().__init__()

        # bottom layers, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True),
            activation_function,
            nn.BatchNorm2d(32)
        )

        # left encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True),
            activation_function,
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=True),
            activation_function,
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=True),
            activation_function,
            nn.BatchNorm2d(256)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=True),
            activation_function,
            nn.BatchNorm2d(512)
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD', bias=True, activation_function=activation_function),
            ResBlock(512, connect_function='ADD', bias=True, activation_function=activation_function),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            activation_function,
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            activation_function,
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            activation_function,
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            activation_function,
            nn.BatchNorm2d(32)
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=0., surrogate_function=surrogate.ATan())

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer and passes through encoder layers
        out_bottom = self.bottom(frame)
        out_conv1 = self.conv1(out_bottom)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv)
        out_add4 = out_deconv4 + out_conv3
        self.Ineurons(self.predict_depth4(out_add4))
        depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 + out_conv2
        self.Ineurons(self.predict_depth3(out_add3))
        depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 + out_conv1
        self.Ineurons(self.predict_depth2(out_add2))
        depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 + out_bottom
        self.Ineurons(self.predict_depth1(out_add1))
        depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4]

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior
