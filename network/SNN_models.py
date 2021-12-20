import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, layer, surrogate

from .blocks import SEWResBlock, NNConvUpsampling, MultiplyBy


class NeuromorphicNet(nn.Module):
    def __init__(self, surrogate_function=surrogate.Sigmoid(), detach_reset=True, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.surrogate_fct = surrogate_function
        self.detach_rst = detach_reset
        self.v_th = v_threshold
        self.v_rst = v_reset

        self.max_test_accuracy = float('inf')
        self.epoch = 0

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                m.v.detach_()
            elif isinstance(m, layer.Dropout):
                m.mask.detach_()

    def get_network_state(self):
        state = []
        for m in self.modules():
            if hasattr(m, 'reset'):
                state.append(m.v)
        return state

    def change_network_state(self, new_state):
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


class StereoSpike(NeuromorphicNet):
    """
    Baseline model, with which we report state-of-the-art performances in the second version of our paper.

    - all neuron potentials must be reset at each timestep
    - predict_depth layers do have biases, but it is equivalent to remove them and reset output I-neurons to the sum
           of all 4 biases, instead of 0.
    """
    def __init__(self, surrogate_function=surrogate.Sigmoid(), detach_reset=True, v_threshold=1.0, v_reset=0.0, multiply_factor=1.):
        super().__init__(surrogate_function=surrogate_function, detach_reset=detach_reset)

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, v_threshold=self.v_th, v_reset=self.v_rst, connect_function='ADD', multiply_factor=multiply_factor),
            SEWResBlock(512, v_threshold=self.v_th, v_reset=self.v_rst, connect_function='ADD', multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
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

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=0.0, surrogate_function=self.surrogate_fct)

    def forward(self, x):

        # x must be of shape [batch_size, num_frames_per_depth_map, 4 (2 cameras - 2 polarities), W, H]
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
        # also output intermediate spike tensors
        return [depth1, depth2, depth3, depth4], [out_rconv, out_add4, out_add3, out_add2, out_add1]

    def calculate_firing_rates(self, x):

        # dictionary to store the firing rates for all layers
        firing_rates_dict = {
            'out_bottom': 0.,
            'out_conv1': 0.,
            'out_conv2': 0.,
            'out_conv3': 0.,
            'out_conv4': 0.,
            'out_rconv': 0.,
            'out_combined': 0.,
            'out_deconv4': 0.,
            'out_add4': 0.,
            'out_deconv3': 0.,
            'out_add3': 0.,
            'out_deconv2': 0.,
            'out_add2': 0.,
            'out_deconv1': 0.,
            'out_add1': 0.,
        }

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer and passes through encoder layers
        out_bottom = self.bottom(frame); firing_rates_dict['out_bottom'] = out_bottom.count_nonzero()/out_bottom.numel()
        out_conv1 = self.conv1(out_bottom); firing_rates_dict['out_conv1'] = out_conv1.count_nonzero()/out_conv1.numel()
        out_conv2 = self.conv2(out_conv1); firing_rates_dict['out_conv2'] = out_conv2.count_nonzero()/out_conv2.numel()
        out_conv3 = self.conv3(out_conv2); firing_rates_dict['out_conv3'] = out_conv3.count_nonzero()/out_conv3.numel()
        out_conv4 = self.conv4(out_conv3); firing_rates_dict['out_conv4'] = out_conv4.count_nonzero()/out_conv4.numel()

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4); firing_rates_dict['out_rconv'] = out_rconv.count_nonzero()/out_rconv.numel()

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv); firing_rates_dict['out_deconv4'] = out_deconv4.count_nonzero()/out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3; firing_rates_dict['out_add4'] = out_add4.count_nonzero()/out_add4.numel()
        self.Ineurons(self.predict_depth4(out_add4))

        out_deconv3 = self.deconv3(out_add4); firing_rates_dict['out_deconv3'] = out_deconv3.count_nonzero()/out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2; firing_rates_dict['out_add3'] = out_add3.count_nonzero()/out_add3.numel()
        self.Ineurons(self.predict_depth3(out_add3))

        out_deconv2 = self.deconv2(out_add3); firing_rates_dict['out_deconv2'] = out_deconv2.count_nonzero()/out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1; firing_rates_dict['out_add2'] = out_add2.count_nonzero()/out_add2.numel()
        self.Ineurons(self.predict_depth2(out_add2))

        out_deconv1 = self.deconv1(out_add2); firing_rates_dict['out_deconv1'] = out_deconv1.count_nonzero()/out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom; firing_rates_dict['out_add1'] = out_add1.count_nonzero()/out_add1.numel()
        self.Ineurons(self.predict_depth1(out_add1))

        return firing_rates_dict

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


class fromZero_feedforward_multiscale_tempo_Matt_SpikeFlowNetLike(NeuromorphicNet):
    """
    Baseline model that we used for our experiments, and whose results are shown in the paper. The difference is the
    use of PLIF neurons instead of IF.
    In our experiments, we used an initial value of tau=3.0 and a multiply_factor of 10.0
    """
    def __init__(self, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, multiply_factor=1.):
        super().__init__(detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
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
            SEWResBlock(512, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau),
            SEWResBlock(512, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau),
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
        return [depth1, depth2, depth3, depth4], [out_rconv, out_add4, out_add3, out_add2, out_add1]

    def calculate_firing_rates(self, x):

        # dictionary to store the firing rates for all layers
        firing_rates_dict = {
            'out_bottom': 0.,
            'out_conv1': 0.,
            'out_conv2': 0.,
            'out_conv3': 0.,
            'out_conv4': 0.,
            'out_rconv': 0.,
            'out_combined': 0.,
            'out_deconv4': 0.,
            'out_add4': 0.,
            'out_deconv3': 0.,
            'out_add3': 0.,
            'out_deconv2': 0.,
            'out_add2': 0.,
            'out_deconv1': 0.,
            'out_add1': 0.,
        }

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer and passes through encoder layers
        out_bottom = self.bottom(frame); firing_rates_dict['out_bottom'] = out_bottom.count_nonzero()/out_bottom.numel()
        out_conv1 = self.conv1(out_bottom); firing_rates_dict['out_conv1'] = out_conv1.count_nonzero()/out_conv1.numel()
        out_conv2 = self.conv2(out_conv1); firing_rates_dict['out_conv2'] = out_conv2.count_nonzero()/out_conv2.numel()
        out_conv3 = self.conv3(out_conv2); firing_rates_dict['out_conv3'] = out_conv3.count_nonzero()/out_conv3.numel()
        out_conv4 = self.conv4(out_conv3); firing_rates_dict['out_conv4'] = out_conv4.count_nonzero()/out_conv4.numel()

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4); firing_rates_dict['out_rconv'] = out_rconv.count_nonzero()/out_rconv.numel()

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv); firing_rates_dict['out_deconv4'] = out_deconv4.count_nonzero()/out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3; firing_rates_dict['out_add4'] = out_add4.count_nonzero()/out_add4.numel()
        self.Ineurons(self.predict_depth4(out_add4))

        out_deconv3 = self.deconv3(out_add4); firing_rates_dict['out_deconv3'] = out_deconv3.count_nonzero()/out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2; firing_rates_dict['out_add3'] = out_add3.count_nonzero()/out_add3.numel()
        self.Ineurons(self.predict_depth3(out_add3))

        out_deconv2 = self.deconv2(out_add3); firing_rates_dict['out_deconv2'] = out_deconv2.count_nonzero()/out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1; firing_rates_dict['out_add2'] = out_add2.count_nonzero()/out_add2.numel()
        self.Ineurons(self.predict_depth2(out_add2))

        out_deconv1 = self.deconv1(out_add2); firing_rates_dict['out_deconv1'] = out_deconv1.count_nonzero()/out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom; firing_rates_dict['out_add1'] = out_add1.count_nonzero()/out_add1.numel()
        self.Ineurons(self.predict_depth1(out_add1))

        return firing_rates_dict

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


class fromZero_feedforward_multiscale_tempo_monocular_SpikeFlowNetLike(NeuromorphicNet):
    """
    This model only takes the data from 1 camera, hence only two channels in the initial 'bottom' convolution.
    """
    def __init__(self, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity, multiply_factor=1.):
        super().__init__(detach_reset=detach_reset)

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
            SEWResBlock(512, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau),
            SEWResBlock(512, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor, use_plif=True, tau=tau),
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
        return [depth1, depth2, depth3, depth4]#, [out_rconv, out_add4, out_add3, out_add2, out_add1]

    def calculate_firing_rates(self, x):

        # dictionary to store the firing rates for all layers
        firing_rates_dict = {
            'out_bottom': 0.,
            'out_conv1': 0.,
            'out_conv2': 0.,
            'out_conv3': 0.,
            'out_conv4': 0.,
            'out_rconv': 0.,
            'out_combined': 0.,
            'out_deconv4': 0.,
            'out_add4': 0.,
            'out_deconv3': 0.,
            'out_add3': 0.,
            'out_deconv2': 0.,
            'out_add2': 0.,
            'out_deconv1': 0.,
            'out_add1': 0.,
        }

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer and passes through encoder layers
        out_bottom = self.bottom(frame); firing_rates_dict['out_bottom'] = out_bottom.count_nonzero()/out_bottom.numel()
        out_conv1 = self.conv1(out_bottom); firing_rates_dict['out_conv1'] = out_conv1.count_nonzero()/out_conv1.numel()
        out_conv2 = self.conv2(out_conv1); firing_rates_dict['out_conv2'] = out_conv2.count_nonzero()/out_conv2.numel()
        out_conv3 = self.conv3(out_conv2); firing_rates_dict['out_conv3'] = out_conv3.count_nonzero()/out_conv3.numel()
        out_conv4 = self.conv4(out_conv3); firing_rates_dict['out_conv4'] = out_conv4.count_nonzero()/out_conv4.numel()

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4); firing_rates_dict['out_rconv'] = out_rconv.count_nonzero()/out_rconv.numel()

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv); firing_rates_dict['out_deconv4'] = out_deconv4.count_nonzero()/out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3; firing_rates_dict['out_add4'] = out_add4.count_nonzero()/out_add4.numel()
        self.Ineurons(self.predict_depth4(out_add4))

        out_deconv3 = self.deconv3(out_add4); firing_rates_dict['out_deconv3'] = out_deconv3.count_nonzero()/out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2; firing_rates_dict['out_add3'] = out_add3.count_nonzero()/out_add3.numel()
        self.Ineurons(self.predict_depth3(out_add3))

        out_deconv2 = self.deconv2(out_add3); firing_rates_dict['out_deconv2'] = out_deconv2.count_nonzero()/out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1; firing_rates_dict['out_add2'] = out_add2.count_nonzero()/out_add2.numel()
        self.Ineurons(self.predict_depth2(out_add2))

        out_deconv1 = self.deconv1(out_add2); firing_rates_dict['out_deconv1'] = out_deconv1.count_nonzero()/out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom; firing_rates_dict['out_add1'] = out_add1.count_nonzero()/out_add1.numel()
        self.Ineurons(self.predict_depth1(out_add1))

        return firing_rates_dict

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


####################
# ABLATION STUDIES #
####################

class fromZero_feedforward_multiscale_tempo_Matt_noskip_SpikeFlowNetLike(NeuromorphicNet):
    """
    model that makes intermediate predictions at different times thanks to a pool of neuron that is common.

    IT IS FULLY SPIKING AND HAS THE BEST MDE WE'VE SEEN SO FAR !!!

    """
    def __init__(self, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity, multiply_factor=1.):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
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
        # NO SKIP CONNECTION
        out_deconv4 = self.deconv4(out_rconv)
        out_add4 = out_deconv4 #+ out_conv3
        self.Ineurons(self.predict_depth4(out_add4))
        depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)
        out_add3 = out_deconv3 #+ out_conv2
        self.Ineurons(self.predict_depth3(out_add3))
        depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)
        out_add2 = out_deconv2 #+ out_conv1
        self.Ineurons(self.predict_depth2(out_add2))
        depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)
        out_add1 = out_deconv1 #+ out_bottom
        self.Ineurons(self.predict_depth1(out_add1))
        depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        return [depth1, depth2, depth3, depth4]

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior


class fromZero_feedforward_multiscale_tempo_Matt_cutpredict_SpikeFlowNetLike(NeuromorphicNet):
    """
    Removed deepest prediction layer
    """
    def __init__(self, use_plif=False, detach_reset=True, tau=10., v_threshold=1.0, v_reset=0.0, final_activation=nn.Identity, multiply_factor=1.):
        super().__init__(use_plif=use_plif, detach_reset=detach_reset)

        self.is_cext_model = False

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
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
        #self.Ineurons(self.predict_depth4(out_add4))
        #depth4 = self.Ineurons.v

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
        return [depth1, depth2, depth3]#, depth4]

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior
