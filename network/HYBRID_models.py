import torch
from torch import nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, neuron, layer, surrogate
from .blocks import BilinConvUpsampling, NNConvUpsampling, ConvLSTMCell, ResBlock, SEWResBlock, MultiplyBy, SpikingConvLSTMCell


##############
# BASE CLASS #
##############

class HybridNet(nn.Module):
    def __init__(self, use_plif=True, detach_reset=True):
        super().__init__()
        self.use_plif = use_plif
        self.detach_reset = detach_reset

        self.is_cext_model = False

        self.max_test_accuracy = float('inf')
        self.epoch = 0

        self.benchmark = False

    def reset_convLSTM_states(self):
        for m in self.modules():
            if isinstance(m, ConvLSTMCell) or isinstance(m, SpikingConvLSTMCell):
                m.set_hc(None, None)

    def detach(self):
        for m in self.modules():
            if isinstance(m, ConvLSTMCell) or isinstance(m, SpikingConvLSTMCell):
                m.h.detach_()
                m.c.detach_()

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

# TODO: make a binocular network hybrid architecture


######################
# MONOCULAR NETWORKS #
######################

class SpikingConvLSTM_multiscale_concat_Hybrid_SpikeFlowNetLike(HybridNet):
    """
    Fully ANN architecture
    """

    def __init__(self, spiking_bottleneck=False, tau=10., v_threshold=1.0, v_reset=0.0, use_plif=True, detach_reset=True):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False
        self.scales = [(15, 20), (31, 42), (63, 85), (128, 171), (258, 344), (260, 346)]

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # spiking convlstm modules
        self.convlstm1 = SpikingConvLSTMCell(input_channels=64, hidden_channels=64, kernel_size=3, bias=False)
        self.convlstm2 = SpikingConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3, bias=False)
        self.convlstm3 = SpikingConvLSTMCell(input_channels=256, hidden_channels=256, kernel_size=3, bias=False)
        self.convlstm4 = SpikingConvLSTMCell(input_channels=512, hidden_channels=512, kernel_size=3, bias=False)

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(128, 171), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(258, 344), bias=True),
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
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(128, 171), bias=True),
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


class SpikingBottleneck_multiscale_concat_Hybrid_SpikeFlowNetLike(HybridNet):

    def __init__(self, tau=10., v_threshold=1.0, v_reset=0.0, use_plif=True, detach_reset=True):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False
        self.scales = [(15, 20), (31, 42), (63, 85), (128, 171), (258, 344), (260, 346)]

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.),
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(128, 171), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(258, 344), bias=True),
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
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(128, 171), bias=True),
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

            # pass through stateful encoder layers
            out_conv1 = self.conv1(out_bottom)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

            # pass through residual blocks
            out_rconv = self.bottleneck(out_conv4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_conv3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_conv2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_conv1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class accumulator_multiscale_concat_Hybrid_SpikeFlowNetLike(HybridNet):
    """
    Accumulates spikes (ints)
    """

    def __init__(self, spiking_bottleneck=False, tau=10., v_threshold=1.0, v_reset=0.0, use_plif=True, detach_reset=True):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False
        self.scales = [(15, 20), (31, 42), (63, 85), (128, 171), (258, 344), (260, 346)]

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(128, 171), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(258, 344), bias=True),
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
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(128, 171), bias=True),
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

        # variables to accumulate spikes
        out_mem_bottom = 0
        out_mem1 = 0
        out_mem2 = 0
        out_mem3 = 0
        out_mem4 = 0

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        for t in range(x.shape[1]):
            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)
            out_mem_bottom += out_bottom

            # pass through stateful encoder layers
            out_conv1 = self.conv1(out_bottom)
            out_mem1 += out_conv1
            out_conv2 = self.conv2(out_conv1)
            out_mem2 += out_conv2
            out_conv3 = self.conv3(out_conv2)
            out_mem3 += out_conv3
            out_conv4 = self.conv4(out_conv3)
            out_mem4 += out_conv4

        # pass through residual blocks
        out_rconv = self.bottleneck(out_mem4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_mem3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_mem2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_mem1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_mem_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class maxpool_multiscale_concat_Hybrid_SpikeFlowNetLike(HybridNet):
    """
    In this model, the downsampling is done by Maxpool layers
    """

    def __init__(self, tau=10., v_threshold=1.0, v_reset=0.0, use_plif=True, detach_reset=True):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False
        self.scales = [(11, 16), (24, 34), (57, 78), (123, 166), (256, 342), (260, 346)]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset),
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        # residual layers
        # TODO: give an option to choose between ANN or SNN bottleneck
        #  e.g. argument 'spiking_bottleneck' = True by default
        #  --> SEWResBlock(512, connect_function='ADD') if spiking_bottleneck else ResBlock(512, connect_function='ADD')
        self.bottleneck = nn.Sequential(
            ResBlock(512, connect_function='ADD', bias=True),
            ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=512, kernel_size=3, up_size=(24, 34), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(512)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=256, kernel_size=3, up_size=(57, 78), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=128, kernel_size=3, up_size=(123, 166), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=64, kernel_size=3, up_size=(256, 342), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=512 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=32, kernel_size=3, up_size=(24, 34), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(57, 78), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(123, 166), bias=True),
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

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):
            frame = x[:, t, :, :, :]

            out_conv1 = self.conv1(frame)
            out_maxpool1 = self.maxpool1(out_conv1)
            out_conv2 = self.conv2(out_maxpool1)
            out_maxpool2 = self.maxpool2(out_conv2)
            out_conv3 = self.conv3(out_maxpool2)
            out_maxpool3 = self.maxpool1(out_conv3)
            out_conv4 = self.conv4(out_maxpool3)
            out_maxpool4 = self.maxpool4(out_conv4)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_maxpool4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_conv4), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_conv3), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_conv2), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_conv1), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class multiscale_concat_Hybrid_SpikeFlowNetLike(HybridNet):
    """
    Fully ANN architecture
    """

    def __init__(self, spiking_bottleneck=False, tau=10., v_threshold=1.0, v_reset=0.0, use_plif=True, detach_reset=True):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        if use_plif:
            SpikingNode = neuron.ParametricLIFNode
        else:
            SpikingNode = neuron.LIFNode

        self.scales = [(15, 20), (31, 42), (63, 85), (128, 171), (258, 344), (260, 346)]

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset) if use_plif else neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(128, 171), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(258, 344), bias=True),
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
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(128, 171), bias=True),
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

            # pass through stateful encoder layers
            out_conv1 = self.conv1(out_bottom)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_conv3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_conv2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_conv1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


##########################
# "FEEDFORWARD" NETWORKS #
##########################


class heaviside_feedforward_multiscale_concat_Hybrid_SpikeFlowNetLike(HybridNet):
    """
    This network is equivalent as having an ANN decoder whose activation functions (except for prediction layers) are
    Heaviside Step functions with surrogate gradient.
    A LIF layer with tau=1 emulates this behaviour and is stateless.
    """

    def __init__(self, spiking_bottleneck=False, tau=10., v_threshold=1.0, v_reset=0.0, use_plif=False, detach_reset=True):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False
        self.scales = [(15, 20), (31, 42), (63, 85), (128, 171), (258, 344), (260, 346)]

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(63, 85), bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(128, 171), bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(258, 344), bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2 + 1, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2 + 1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2 + 1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset), #nn.LeakyReLU(0.1),
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
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(128, 171), bias=True),
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

        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = self.bottom(frame)

        # pass through stateful encoder layers
        out_conv1 = self.conv1(out_bottom)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_conv3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_conv2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_conv1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4


class feedforward_multiscale_concat_Hybrid_SpikeFlowNetLike(HybridNet):
    """
    Input is only 1 frame of a large time window, so spiking neurons could be IFs instead of LIFs as well
    """

    def __init__(self, spiking_bottleneck=False, tau=10., v_threshold=1.0, v_reset=0.0, use_plif=False, detach_reset=True):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False
        self.scales = [(15, 20), (31, 42), (63, 85), (128, 171), (258, 344), (260, 346)]

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
            SEWResBlock(512, connect_function='ADD', use_plif=False, tau=tau, v_threshold=1., v_reset=0., multiply_factor=15.) if spiking_bottleneck else ResBlock(512, connect_function='ADD', bias=True),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128)
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(128, 171), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(258, 344), bias=True),
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
            NNConvUpsampling(in_channels=256, out_channels=32, kernel_size=3, up_size=(63, 85), bias=True),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=32, kernel_size=3, up_size=(128, 171), bias=True),
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

            # pass through stateful encoder layers
            out_conv1 = self.conv1(out_bottom)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # latent representation is then decoded by stateless upsampling layers
        out_depth4 = self.predict_depth4(out_rconv)
        out_deconv4 = self.deconv4(out_rconv)
        out_cat4 = torch.cat((out_depth4, out_deconv4, out_conv3), dim=1)
        out_cat4 = self.resconv4(out_cat4)

        out_depth3 = self.predict_depth3(out_cat4)
        out_deconv3 = self.deconv3(out_cat4)
        out_cat3 = torch.cat((out_depth3, out_deconv3, out_conv2), dim=1)
        out_cat3 = self.resconv3(out_cat3)

        out_depth2 = self.predict_depth2(out_cat3)
        out_deconv2 = self.deconv2(out_cat3)
        out_cat2 = torch.cat((out_depth2, out_deconv2, out_conv1), dim=1)
        out_cat2 = self.resconv2(out_cat2)

        out_deconv1 = self.deconv1(out_cat2)
        out_cat1 = torch.cat((out_deconv1, out_bottom), dim=1)
        out_cat1 = self.resconv1(out_cat1)

        out_depth1 = self.predict_depth1(out_cat1)

        return out_depth1, out_depth2, out_depth3, out_depth4

