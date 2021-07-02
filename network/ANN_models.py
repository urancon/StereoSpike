import torch
from torch import nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, neuron, layer, surrogate
from .blocks import BilinConvUpsampling, NNConvUpsampling, ConvLSTMCell, ResBlock, MultiplyBy, AttentionGate


##############
# BASE CLASS #
##############

class AnalogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_test_accuracy = float('inf')
        self.epoch = 0

    def reset_convLSTM_states(self):
        for m in self.modules():
            if isinstance(m, ConvLSTMCell):
                m.set_hc(None, None)

    def detach(self):
        for m in self.modules():
            if isinstance(m, ConvLSTMCell):
                m.h.detach()
                m.c.detach()

    def increment_epoch(self):
        self.epoch += 1

    def get_max_accuracy(self):
        return self.max_test_accuracy

    def update_max_accuracy(self, new_acc):
        self.max_test_accuracy = new_acc

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


######################
# BINOCULAR NETWORKS #
######################

class biased_binocular_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        # left bottom layer, preprocessing the input spike frame without downsampling
        self.bottomL = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # right bottom layer, preprocessing the input spike frame without downsampling
        self.bottomR = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # left encoder layers (downsampling)
        self.conv1L = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2L = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3L = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4L = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # left encoder layers (downsampling)
        self.conv1R = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2R = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3R = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4R = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # left reccurrent convlstm layers in encoders
        self.convlstm1L = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2L = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3L = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4L = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # right reccurrent convlstm layers in encoders
        self.convlstm1R = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2R = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3R = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4R = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(1024, connect_function='ADD', bias=True),
            ResBlock(1024, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=1024, out_channels=512, kernel_size=3, up_size=(35, 46), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(67, 89), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(132, 175), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(262, 348), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 512 + 1, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 256 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 128 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2 + 64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=1024, out_channels=32, kernel_size=3, up_size=(35, 46), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(67, 89), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(132, 175), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(260, 346), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, xL, xR):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(xL.shape[1]):

            # left and right data
            frameL = xL[:, t, :, :, :]
            frameR = xR[:, t, :, :, :]

            # left encoder pathway
            out_bottomL = self.bottomL(frameL)
            out_convlstm1L = self.convlstm1L(self.conv1L(out_bottomL))
            out_convlstm2L = self.convlstm2L(self.conv2L(out_convlstm1L))
            out_convlstm3L = self.convlstm3L(self.conv3L(out_convlstm2L))
            out_convlstm4L = self.convlstm4L(self.conv4L(out_convlstm3L))

            # right encoder pathway
            out_bottomR = self.bottomR(frameR)
            out_convlstm1R = self.convlstm1R(self.conv1R(out_bottomR))
            out_convlstm2R = self.convlstm2R(self.conv2R(out_convlstm1R))
            out_convlstm3R = self.convlstm3R(self.conv3R(out_convlstm2R))
            out_convlstm4R = self.convlstm4R(self.conv4R(out_convlstm3R))

        # concatenate the output of both encoders, and pass it through residual layers
        out_combined = torch.cat((out_convlstm4L, out_convlstm4R), dim=1)
        out_combined = self.bottleneck(out_combined)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_combined)
        out_deconv4 = self.deconv4(out_combined)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3L, out_convlstm3R), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2L, out_convlstm2R), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1L, out_convlstm1R), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottomL, out_bottomR), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class binocular_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        # left bottom layer, preprocessing the input spike frame without downsampling
        self.bottomL = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # right bottom layer, preprocessing the input spike frame without downsampling
        self.bottomR = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # left encoder layers (downsampling)
        self.conv1L = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2L = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3L = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4L = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # left encoder layers (downsampling)
        self.conv1R = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2R = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3R = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4R = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # left reccurrent convlstm layers in encoders
        self.convlstm1L = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2L = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3L = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4L = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # right reccurrent convlstm layers in encoders
        self.convlstm1R = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2R = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3R = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4R = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(1024, connect_function='ADD'),
            ResBlock(1024, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=1024, out_channels=512, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 512 + 1, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 256 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 128 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2 + 64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=1024, out_channels=32, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(260, 346)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, xL, xR):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(xL.shape[1]):

            # left and right data
            frameL = xL[:, t, :, :, :]
            frameR = xR[:, t, :, :, :]

            # left encoder pathway
            out_bottomL = self.bottomL(frameL)
            out_convlstm1L = self.convlstm1L(self.conv1L(out_bottomL))
            out_convlstm2L = self.convlstm2L(self.conv2L(out_convlstm1L))
            out_convlstm3L = self.convlstm3L(self.conv3L(out_convlstm2L))
            out_convlstm4L = self.convlstm4L(self.conv4L(out_convlstm3L))

            # right encoder pathway
            out_bottomR = self.bottomR(frameR)
            out_convlstm1R = self.convlstm1R(self.conv1R(out_bottomR))
            out_convlstm2R = self.convlstm2R(self.conv2R(out_convlstm1R))
            out_convlstm3R = self.convlstm3R(self.conv3R(out_convlstm2R))
            out_convlstm4R = self.convlstm4R(self.conv4R(out_convlstm3R))

        # concatenate the output of both encoders, and pass it through residual layers
        out_combined = torch.cat((out_convlstm4L, out_convlstm4R), dim=1)
        out_combined = self.bottleneck(out_combined)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_combined)
        out_deconv4 = self.deconv4(out_combined)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3L, out_convlstm3R), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2L, out_convlstm2R), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1L, out_convlstm1R), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottomL, out_bottomR), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class binocular_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike_v2(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        # left bottom layer, preprocessing the input spike frame without downsampling
        self.bottomL = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # right bottom layer, preprocessing the input spike frame without downsampling
        self.bottomR = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # left encoder layers (downsampling)
        self.conv1L = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2L = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3L = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4L = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # left encoder layers (downsampling)
        self.conv1R = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2R = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3R = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4R = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # left reccurrent convlstm layers in encoders
        self.convlstm1L = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2L = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3L = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4L = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # right reccurrent convlstm layers in encoders
        self.convlstm1R = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2R = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3R = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4R = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(1024, connect_function='ADD'),
            ResBlock(1024, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=1024, out_channels=512, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 512 + 1, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 256 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 128 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2 + 64 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=1024, out_channels=32, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth0 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(260, 346)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, xL, xR):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(xL.shape[1]):

            # left and right data
            frameL = xL[:, t, :, :, :]
            frameR = xR[:, t, :, :, :]

            # left encoder pathway
            out_bottomL = self.bottomL(frameL)
            out_convlstm1L = self.convlstm1L(self.conv1L(out_bottomL))
            out_convlstm2L = self.convlstm2L(self.conv2L(out_convlstm1L))
            out_convlstm3L = self.convlstm3L(self.conv3L(out_convlstm2L))
            out_convlstm4L = self.convlstm4L(self.conv4L(out_convlstm3L))

            # right encoder pathway
            out_bottomR = self.bottomR(frameR)
            out_convlstm1R = self.convlstm1R(self.conv1R(out_bottomR))
            out_convlstm2R = self.convlstm2R(self.conv2R(out_convlstm1R))
            out_convlstm3R = self.convlstm3R(self.conv3R(out_convlstm2R))
            out_convlstm4R = self.convlstm4R(self.conv4R(out_convlstm3R))

        # concatenate the output of both encoders, and pass it through residual layers
        out_combined = torch.cat((out_convlstm4L, out_convlstm4R), dim=1)
        out_combined = self.bottleneck(out_combined)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_combined)
        out_deconv4 = self.deconv4(out_combined)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3L, out_convlstm3R), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2L, out_convlstm2R), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1L, out_convlstm1R), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_depth1 = self.predict_depth1(out_cat2)
        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_depth1, out_deconv1, out_bottomL, out_bottomR), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        out_depth0 = self.predict_depth0(out_cat1)

        return out_depth0, out_depth1, out_depth2, out_depth3, out_depth4


######################
# MONOCULAR NETWORKS #
######################

class IFoutput_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD'),
            ResBlock(512, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            neuron.IFNode(v_threshold=float('inf'), v_reset=0., surrogate_function=surrogate.ATan()),
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            neuron.IFNode(v_threshold=float('inf'), v_reset=0., surrogate_function=surrogate.ATan()),
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            neuron.IFNode(v_threshold=float('inf'), v_reset=0., surrogate_function=surrogate.ATan()),
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            neuron.IFNode(v_threshold=float('inf'), v_reset=0., surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4

    def set_init_depths_potentials(self, groundtruth):
        scales = [(260, 346), (132, 175), (67, 89), (35, 46)]
        scale_idx = 0
        for m in self.modules():
            if isinstance(m, neuron.IFNode):
                scale = scales[scale_idx]
                rescaled_pots = F.interpolate(groundtruth, size=scale, mode='bilinear', align_corners=False)
                m.v = rescaled_pots


class attention_biased_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD', bias=True),
            ResBlock(512, connect_function='ADD', bias=True),
        )

        # attention gates
        self.attention1 = AttentionGate(g_channels=64, x_channels=32, out_size=(262, 348))
        self.attention2 = AttentionGate(g_channels=128, x_channels=64, out_size=(130, 173))
        self.attention3 = AttentionGate(g_channels=256, x_channels=128, out_size=(64, 86))
        self.attention4 = AttentionGate(g_channels=512, x_channels=256, out_size=(31, 42))

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(64, 86), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(130, 173), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(31, 42), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(64, 86), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(130, 173), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_attention4 = self.attention4(out_rconv, out_convlstm3)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_attention4), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_attention3 = self.attention3(out_cat4, out_convlstm2)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_attention3), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_attention2 = self.attention2(out_cat3, out_convlstm1)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_attention2), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_attention1 = self.attention1(out_cat2, out_bottom)
        out_cat1 = torch.cat((out_deconv1, out_attention1), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class sigmoid_biased_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD', bias=True),
            ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(35, 46), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
            MultiplyBy(10)
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(67, 89), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
            MultiplyBy(10)
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(132, 175), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
            MultiplyBy(10)
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
            MultiplyBy(10)
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class biased_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD', bias=True),
            ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(35, 46), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(67, 89), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(132, 175), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class bilinear_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD'),
            ResBlock(512, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            BilinConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            BilinConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            BilinConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            BilinConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            BilinConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            BilinConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            BilinConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            BilinConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD'),
            ResBlock(512, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_convlstm3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_convlstm2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_convlstm1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class noreinjection_multiscale_concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD'),
            ResBlock(512, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_deconv4, out_convlstm3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_deconv3, out_convlstm2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_deconv2, out_convlstm1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)
        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class concat_Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD'),
            ResBlock(512, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_deconv4, out_convlstm3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_deconv3, out_convlstm2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_deconv2, out_convlstm1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        return self.predict_depth(out_cat1)


class Analog_ConvLSTM_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # reccurrent convlstm layers in encoders
        self.convlstm1 = ConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3)
        self.convlstm4 = ConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3)

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD'),
            ResBlock(512, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # top convolutional layer, processing the output of the decoder and the output of the bottom layer
        self.top = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_convlstm1 = self.convlstm1(out_conv1)
            out_conv2 = self.conv2(out_convlstm1)
            out_convlstm2 = self.convlstm2(out_conv2)
            out_conv3 = self.conv3(out_convlstm2)
            out_convlstm3 = self.convlstm3(out_conv3)
            out_conv4 = self.conv4(out_convlstm3)
            out_convlstm4 = self.convlstm4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_convlstm4)

        # latent representation is then decoded by stateless upsampling layers
        out_deconv4 = self.deconv4(out_rconv)
        out_add4 = out_deconv4 + out_convlstm3

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 + out_convlstm2

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 + out_convlstm1

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 + out_bottom

        # final processing and update of output depth potentials
        out_top = self.top(out_add1)

        return self.predict_depth(out_top)


class Analog_feedforward_SpikeFlowNetLike(AnalogNet):
    """
    Fully ANN architecture
    """
    def __init__(self):
        super().__init__()

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD'),
            ResBlock(512, connect_function='ADD'),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(35, 46)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(67, 89)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(132, 175)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(262, 348)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through stateful encoder layers: strided convolutions + convlstms
            out_conv1 = self.conv1(out_bottom)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # latent representation is then decoded by stateless upsampling layers
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_deconv4, out_conv3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_deconv3, out_conv2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_deconv2, out_conv1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        return self.predict_depth(out_cat1)

