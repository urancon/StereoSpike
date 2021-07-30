import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, layer, surrogate

from .blocks import SEWResBlock, OneToOne, NNConvUpsampling, MultiplyBy, SpikingConvLSTMCell

# Energy costs of Accumulation and Multiplication-Accumulation according to Fusion-FlowNet
ENERGY_COST_ACC = 0.9  # pJ
ENERGY_COST_MACC = 4.6  # pJ


##############
# BASE CLASS #
##############

class NeuromorphicNet(nn.Module):
    def __init__(self, T=None, use_plif=True, detach_reset=True):
        super().__init__()
        self.T = T
        self.use_plif = use_plif
        self.detach_reset = detach_reset

        self.is_cext_model = False

        self.max_test_accuracy = float('inf')
        self.epoch = 0

        self.benchmark = False

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                m.v.detach_()
            elif isinstance(m, layer.Dropout):
                m.mask.detach_()

    def reset_convLSTM_states(self):
        for m in self.modules():
            if isinstance(m, SpikingConvLSTMCell):
                m.set_hc(None, None)

    def get_state(self):
        state = []
        for m in self.modules():
            if hasattr(m, 'reset'):
                state.append(m.v)
        return state

    def change_state(self, new_state):
        module_index = 0
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.v = new_state[module_index]
                module_index += 1

    def set_output_potentials(self, new_pots):
        module_index = 0
        for m in self.modules():
            if isinstance(m, neuron.IFNode):
                m.v = new_pots[module_index]
                module_index += 1

    def increment_epoch(self):
        self.epoch += 1

    def get_max_accuracy(self):
        return self.max_test_accuracy

    def update_max_accuracy(self, new_acc):
        self.max_test_accuracy = new_acc

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def benchmark(self, b: bool):
        self.benchmark = b

    def energy_consumption_SNN(self):
        # TODO: DONT FORGET SKIP CONNECTIONS !!! THEY ARE A SOURCE OF SYNAPTIC OPERATIONS !!!
        #  --> we have to count twice the synaptic operations from downsampling convolutional layers (not exactly)
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, OneToOne):
                print(m)
        # TODO: continuer l'étude

        E_total = 0
        for l in range(n_layer):
            N_ops += n_synaptic_ops * firing_rates * ENERGY_COST_AC
        E_total *= self.T

        return E_total
        '''
        raise NotImplementedError

    def energy_consumption_ANN(self):
        raise NotImplementedError


##############
# NETWORKS   #
##############

class fromZero_concat_SpikeFlowNetLike(NeuromorphicNet):
    """
    Fully spiking architecture, with output depth potentials initialized at 0 everywhere before forward passing input
    data sample.
    So the prediction is made from scratch. To train with shuffled procedure. As a result, output potentials are not
    permanent.
    """
    def __init__(self, T=None, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=15.),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=15.),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(31, 42), bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(63, 85), bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(128, 171), bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=3, up_size=(258, 344), bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # convolutions in between upsampling layers
        self.resconv4 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.resconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128 * 2, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.resconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.resconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * 2, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            MultiplyBy(15.),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=32, kernel_size=3, up_size=(260, 346), bias=False),
            MultiplyBy(15.),
            OneToOne((32, 260, 346)),
            MultiplyBy(15.),
            neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        self.final_activation = final_activation

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

            self.predict_depth(out_cat1)

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [self.predict_depth[-1].v]  # OR return self.final_activation(self.predict_depth[-1].v)

    def set_init_depths_potentials(self, depth_prior):
        self.predict_depth[-1].v = depth_prior


class fromZero_feedforward_multiscale_SpikeFlowNetLike(NeuromorphicNet):
    """
    model that makes intermediate predictions at different times thanks to a pool of neuron that is common.

    IT IS FULLY SPIKING AND HAS THE BEST MDE WE'VE SEEN SO FAR !!!

    """
    def __init__(self, T=None, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity, multiply_factor=1.):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),

            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )
        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())

        self.final_activation = final_activation

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = self.bottom(frame)

        # pass through encoder layers
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


class fromZero_SpikeFlowNetLike(NeuromorphicNet):
    """
    Fully spiking architecture, with output depth potentials initialized at 0 everywhere before forward passing input
    data sample.
    So the prediction is made from scratch. To train with shuffled procedure. As a result, output potentials are not
    permanent.
    """
    def __init__(self, T=None, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity, multiply_factor=1.):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.BatchNorm2d(32),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(64, affine=False),
            #nn.BatchNorm2d(64),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(128, affine=False),
            #nn.BatchNorm2d(128),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(256, affine=False),
            #nn.BatchNorm2d(256),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(512, affine=False),
            #nn.BatchNorm2d(512),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif, multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # top convolutional layer, processing the output of the decoder and the output of the bottom layer
        self.top = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.BatchNorm2d(32),
            MultiplyBy(multiply_factor),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            OneToOne((32, 260, 346)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=float('inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        self.final_activation = final_activation

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through encoder layers
            out_conv1 = self.conv1(out_bottom)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

            # pass through residual blocks
            out_rconv = self.bottleneck(out_conv4)

            # gradually upsample while concatenating and passing through skip connections
            out_deconv4 = self.deconv4(out_rconv)
            out_add4 = out_deconv4 + out_conv3

            out_deconv3 = self.deconv3(out_add4)
            out_add3 = out_deconv3 + out_conv2

            out_deconv2 = self.deconv2(out_add3)
            out_add2 = out_deconv2 + out_conv1

            out_deconv1 = self.deconv1(out_add2)
            out_add1 = out_deconv1 + out_bottom

            # final processing and update of output depth potentials
            out_top = self.top(out_add1)
            self.predict_depth(out_top)

            #self.predict_depth[-1].v = self.final_activation(self.predict_depth[-1].v)

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [self.predict_depth[-1].v]
        #return self.final_activation(self.predict_depth[-1].v)

    def set_init_depths_potentials(self, depth_prior):
        self.predict_depth[-1].v = depth_prior


class SpikeFlowNetLike(NeuromorphicNet):
    """
    A fully spiking Spike-FlowNet network, but for monocular depth estimation.
    Its architecture is very much like a U-Net, and it outputs a single-channel depth map of the same size as the input
    field of view.
    Basically, we have replaced all ReLU activation functions in Spike-FlowNet by LIFNodes -> non-linearity
    Intermediate depth maps (see paper) come out of IFNodes with an infinite threshold.

    UPDATE: added a head layer before encoders, and encoders now use a 5x5 kernel like in "estimating monocular dense
     depth from events paper", whereas "spikeflownet" paper used 3x3 kernels.
     Even NNConvUpsampling layers use 5x5 kernels
    """
    def __init__(self, T=None, use_plif=True, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf'), final_activation=nn.Identity):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.BatchNorm2d(32),
            MultiplyBy(),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(64, affine=False),
            #nn.BatchNorm2d(64),
            MultiplyBy(),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(128, affine=False),
            #nn.BatchNorm2d(128),
            MultiplyBy(),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(256, affine=False),
            #nn.BatchNorm2d(256),
            MultiplyBy(),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(512, affine=False),
            #nn.BatchNorm2d(512),
            MultiplyBy(),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # top convolutional layer, processing the output of the decoder and the output of the bottom layer
        self.top = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.BatchNorm2d(32),
            MultiplyBy(),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            OneToOne((32, 260, 346)),
            MultiplyBy(),
            neuron.IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        self.final_activation = final_activation

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x.shape[1]):

            frame = x[:, t, :, :, :]

            # data is fed in through the bottom layer
            out_bottom = self.bottom(frame)

            # pass through encoder layers
            out_conv1 = self.conv1(out_bottom)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)

            # pass through residual blocks
            out_rconv = self.bottleneck(out_conv4)

            # gradually upsample while concatenating and passing through skip connections
            out_deconv4 = self.deconv4(out_rconv)
            out_add4 = out_deconv4 + out_conv3

            out_deconv3 = self.deconv3(out_add4)
            out_add3 = out_deconv3 + out_conv2

            out_deconv2 = self.deconv2(out_add3)
            out_add2 = out_deconv2 + out_conv1

            out_deconv1 = self.deconv1(out_add2)
            out_add1 = out_deconv1 + out_bottom

            # final processing and update of output depth potentials
            out_top = self.top(out_add1)
            self.predict_depth(out_top)

            #self.predict_depth[-1].v = self.final_activation(self.predict_depth[-1].v)

        # the membrane potentials of the output IF neuron carry the depth prediction
        #return self.predict_depth[-1].v
        return self.final_activation(self.predict_depth[-1].v)

        '''
        if self.benchmark:
            output_pots = self.final_activation(self.predict_depth[-1].v)
            firing_rates = [out_bottom,
                            out_conv1, out_conv2, out_conv3, out_conv4,
                            out_rconv,
                            out_deconv4, out_deconv3, out_deconv2, out_deconv1,
                            out_top]
            firing_rates = [sum(elem) / (elem[1] * elem[2] * elem[3]) for elem in firing_rates]
            return output_pots, firing_rates
        '''

    def set_init_depths_potentials(self, depth_prior):
        self.predict_depth[-1].v = depth_prior


class FusionFlowNetLike(NeuromorphicNet):
    """
    A fully spiking Fusion-FlowNet network, but for binocular depth estimation.
    Its architecture is very much like a U-Net, and it outputs a single-channel depth map of the same size as the input
    field of view. The main difference with a U-Net is in the encoder, which takes as inputs 2 DVS spike trains.
    Basically, we have replaced all ReLU activation functions in Fusion-FlowNet by LIFNodes -> non-linearity.
    The stream of spikes of the second DVS camera replaces the grayscale image that is fed to the network in
    Fusion-FlowNet. Concatenations are also replaced by additions.
    Intermediate depth maps (see paper) come out of IFNodes with an infinite threshold.
    """
    def __init__(self, T=None, use_plif=True, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf'), final_activation=nn.Identity):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layers, preprocessing the left input spike frame without downsampling
        self.bottom_left = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # bottom layer, preprocessing the right input spike frame without downsampling
        self.bottom_right = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling) for the left input spiketrain
        self.conv1_left = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2_left = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3_left = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4_left = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling) for the right input spiketrain
        self.conv1_right = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2_right = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3_right = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4_right = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # top convolutional layer, processing the output of the decoder and the output of the bottom layer
        self.top = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            neuron.ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            OneToOne((32, 260, 346)),
            neuron.IFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        self.final_activation = final_activation

    def forward(self, x_left, x_right):
        # x_left and x_right must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        for t in range(x_left.shape[1]):

            # separate left and right input spiketrains
            left_frame = x_left[:, t, :, :, :]
            right_frame = x_right[:, t, :, :, :]

            out_bottom_left = self.bottom_left(left_frame)
            out_bottom_right = self.bottom_right(right_frame)

            # pass through left encoder layers
            out_conv1_left = self.conv1_left(out_bottom_left)
            out_conv2_left = self.conv2_left(out_conv1_left)
            out_conv3_left = self.conv3_left(out_conv2_left)
            out_conv4_left = self.conv4_left(out_conv3_left)

            # pass through right encoder layers
            out_conv1_right = self.conv1_right(out_bottom_right)
            out_conv2_right = self.conv2_right(out_conv1_right)
            out_conv3_right = self.conv3_right(out_conv2_right)
            out_conv4_right = self.conv4_right(out_conv3_right)

            # concatenate the features calculated by both pathways along channel dimension
            # out_conv4 = torch.cat((out_conv4_left, out_conv4_right), dim=1)
            out_conv4 = out_conv4_left + out_conv4_right

            # pass through residual blocks
            out_rconv = self.bottleneck(out_conv4)

            # gradually upsample while summing spikes and passing through skip connections
            out_deconv4 = self.deconv4(out_rconv)
            out_add4 = out_deconv4 + out_conv3_left + out_conv3_right

            out_deconv3 = self.deconv3(out_add4)
            out_add3 = out_deconv3 + out_conv2_left + out_conv2_right

            out_deconv2 = self.deconv2(out_add3)
            out_add2 = out_deconv2 + out_conv1_left + out_conv1_right

            out_deconv1 = self.deconv1(out_add2)
            out_add1 = out_deconv1 + out_bottom_left + out_bottom_right

            # final processing and update of output depth potentials
            out_top = self.top(out_add1)
            self.predict_depth(out_top)

        # the membrane potentials of the output IF neuron carry the depth prediction
        return self.final_activation(self.predict_depth[-1].v)

    def set_init_depths_potentials(self, depth_prior):
        self.predict_depth[-1].v = depth_prior


