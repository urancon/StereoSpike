import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from spikingjelly.clock_driven import functional, neuron, layer, surrogate  # , accelerating
from spikingjelly.cext import neuron as cext_neuron

from .plif import ParametricLIFNode, MultiStepParametricLIFNode
#from .blocks import cext_SEWResBlock, SEWResBlock, OneToOne, NNConvUpsampling, AttentionGate, MultiplyBy
from .blocks import SEWResBlock, OneToOne, NNConvUpsampling, AttentionGate, MultiplyBy
from .spiking_convLSTM import SpikingConvLSTMCell

from spikingjelly.clock_driven.neuron import BaseNode


# Energy costs of Accumulation and Multiplication-Accumulation according to Fusion-FlowNet
ENERGY_COST_ACC = 0.9  # pJ
ENERGY_COST_MACC = 4.6  # pJ


class NeuromorphicNet(nn.Module):
    def __init__(self, T=None, use_plif=True, detach_reset=True, detach_input=True):
        super().__init__()
        self.T = T
        self.use_plif = use_plif
        self.detach_reset = detach_reset
        self.detach_input = detach_input

        self.is_cext_model = False

        self.max_test_accuracy = float('inf')
        self.epoch = 0

        self.benchmark = False

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode) or isinstance(m, cext_neuron.BaseNode):
                m.v.detach_()
                """ test
                if isinstance(m, ParametricLIFNode) or isinstance(m, MultiStepParametricLIFNode):
                    m.w.detach_()
                """
            elif isinstance(m, layer.Dropout):
                m.mask.detach_()

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
            if isinstance(m, neuron.IFNode) or isinstance(m, cext_neuron.MultiStepIFNode):
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, OneToOne):
                print(m)
        # TODO: continuer l'étude

        E_total = 0
        for l in range(n_layer):
            N_ops += n_synaptic_ops * firing_rates * ENERGY_COST_AC
        E_total *= self.T

        return E_total

    def energy_consumption_ANN(self):
        raise(NotImplementedError)


class DepthSNN(NeuromorphicNet):
    def __init__(self):
        pass



class SpikeFlowNetLike_cext(NeuromorphicNet):
    """
    A SpikeFlowNetLike network, but with spikingjelly's special CUDA cext acceleration.
    """

    def __init__(self, T=None, use_plif=True, detach_reset=True, detach_input=True, tau=10., v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf'), final_activation=nn.Identity):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset, detach_input=detach_input)

        self.is_cext_model = True

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=(3 - 1) // 2, bias=False),
                nn.BatchNorm2d(64),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(3 - 1) // 2, bias=False),
                nn.BatchNorm2d(128),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(3 - 1) // 2, bias=False),
                nn.BatchNorm2d(256),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=(3 - 1) // 2, bias=False),
                nn.BatchNorm2d(512),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            cext_SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
            cext_SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            layer.SeqToANNContainer(
                NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=3, up_size=(33, 44)),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            layer.SeqToANNContainer(
                NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=3, up_size=(65, 87)),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            layer.SeqToANNContainer(
                NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=3, up_size=(130, 173)),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            layer.SeqToANNContainer(
                NNConvUpsampling(in_channels=64, out_channels=4, kernel_size=3, up_size=(260, 346)),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_input=True, detach_reset=True) if self.use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            layer.SeqToANNContainer(
                # nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(1),
                OneToOne((4, 260, 346)),
            ),
            cext_neuron.MultiStepIFNode(v_threshold=v_infinite_thresh, v_reset=v_reset, surrogate_function='ATan'),
        )

        self.final_activation = final_activation

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]

        # x must be of shape [T, N, C, W, H], so we change it a bit: [N, T, C, W, H] -> [T, N, C, W, H]
        x = x.permute(1, 0, 2, 3, 4)

        # then just forward through the network

        # pass through encoder layers
        out_conv1 = self.conv1(x)
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

        return self.final_activation(self.predict_depth[-1].v)


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
    def __init__(self, T=None, use_plif=True, detach_reset=True, detach_input=True, tau=10., v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf'), final_activation=nn.Identity):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset, detach_input=detach_input)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.BatchNorm2d(32),
            MultiplyBy(),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(64, affine=False),
            #nn.BatchNorm2d(64),
            MultiplyBy(),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(128, affine=False),
            #nn.BatchNorm2d(128),
            MultiplyBy(),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(256, affine=False),
            #nn.BatchNorm2d(256),
            MultiplyBy(),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            #nn.BatchNorm2d(512, affine=False),
            #nn.BatchNorm2d(512),
            MultiplyBy(),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # top convolutional layer, processing the output of the decoder and the output of the bottom layer
        self.top = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(32, affine=False),
            #nn.BatchNorm2d(32),
            MultiplyBy(),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            OneToOne((32, 260, 346)),
            # SpikingInceptionModule(in_channels=32, out_channels=1),
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
    def __init__(self, T=None, use_plif=True, detach_reset=True, detach_input=True, tau=10., v_threshold=1.0, v_reset=0.0, v_infinite_thresh=float('inf'), final_activation=nn.Identity):
        super().__init__(T=T, use_plif=use_plif, detach_reset=detach_reset, detach_input=detach_input)

        self.is_cext_model = False

        # bottom layers, preprocessing the left input spike frame without downsampling
        self.bottom_left = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # bottom layer, preprocessing the right input spike frame without downsampling
        self.bottom_right = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling) for the left input spiketrain
        self.conv1_left = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2_left = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3_left = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4_left = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # encoder layers (downsampling) for the right input spiketrain
        self.conv1_right = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv2_right = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv3_right = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.conv4_right = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
            SEWResBlock(512, tau=tau, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', use_plif=use_plif),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        # top convolutional layer, processing the output of the decoder and the output of the bottom layer
        self.top = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
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

