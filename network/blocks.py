import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from spikingjelly.clock_driven import functional, layer, surrogate
from spikingjelly.clock_driven import neuron
from spikingjelly.cext import neuron as cext_neuron

from .plif import ParametricLIFNode, MultiStepParametricLIFNode
from .csrc.element_wise import SpikesOR


class OneToOne(nn.Module):
    """ Applies an element-wise multiplication between the input tensor and a learnable weight tensor of the same shape.
    Can be interpreted as a one-to-one connection between neurons from an input [W, H] map and an output [W, H] map.

    The need for this emerged in our problem because spikes from the last LIF layer of our networks do not "carry" the
     same depth quantity depending of their position in the field of view. Pixels at the bottom of the image will mostly
     be ground pixels for instance, and there the depth variations are inferior than those on the top of the image.

    UPDATE: it actually turns out not to be a good idea to use such weight connections between the last LIF and IF
     neurons, if there is only one input channel. Proof:

        So we have LIF -> one-to-one -> IF at the end of the SNN.
        Suppose, for a particular output IF neuron at time t, a depth of 10 m to predict with its non-leaking potential.
        The preceding LIF layer could emit 10 spikes, and the one-to-one weights between the two could be of 1 m/spk.
        Therefore, our IF neuron gets a right membrane potential of 10 m. So far so good!

        Now suppose, at the next timestep t+1, a depth to predict of 9 m instead of 10 for the same IF neuron.
        So it should decrease its potential of 1 m.
        But the only way for it to do so is by receiving negative-weighted spikes from the preceding layer.
        For instance, the preceding LIF neuron would emit 1 spike with a one-to-one weight of -1 m/spk.
        But that cannot be, since this weight already equals +1 m/spk !

    SOLUTION: intuitively, we could instead define an N-to-One connection between an input volume with several channels
     and the output depth map. For instance, 2 channels solve the issue above, because such a 2-to-One connection can
     have two weights per output pixel: one for increasing the depth / potential, the other for decreasing it.

    HOWEVER: with fixed update weights and a limited number of timesteps (and therefore of spikes, because there is at
     most 1 spike per timestep) to update the output potential / depth at a pixel, it is impossible to follow depth
     changes greater than the number of timesteps between two labels * the (positive or negative) weight.

    SOLUTION: this hints at prefering N-to-One connections with N >> 2 for more dynamic and accurate depth estimations.
     For example, a 4-to-One connection can have 2 positive and 2 negative weights for a same output pixel / neuron:
     one smaller and one larger for each, thus allowing the translation of larger or finer depth changes.
    """

    __constants__ = ['in_features']
    in_features: tuple
    weight: Tensor

    def __init__(self, in_features: tuple) -> None:
        super(OneToOne, self).__init__()
        self.in_features = in_features

        numel = 1
        for i in in_features:
            numel = numel * i

        self.weight = Parameter(Tensor(numel).reshape(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        return torch.sum(torch.mul(input, self.weight), axis=1)  # axis=1 because we sum along channel dimension

    def extra_repr(self) -> str:
        return 'in_features={}'.format(
            self.in_features is not None
        )


class NNConvUpsampling(nn.Module):
    """
    Upsampling block made to address the production of checkerboard artifacts by transposed convolutions.
    Nearest neighbour (NN) upsampling combined to regular convolutions efficiently gets rid of these patterns.
    Linear interpolation (among others) can produce non-integer values (e.g. 0.5 between 0 and 1), which go against the
    philosophy of spiking neural networks, which receive integer amounts of input spikes. Having noticed that, NN
    upsampling is compatible with SNN's philosophy and could be implemented on dedicated hardware.

    See https://distill.pub/2016/deconv-checkerboard/ for more insights on this phenomenon.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple):
        super(NNConvUpsampling, self).__init__()

        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.up(x)
        return out


class cext_SEWResBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) residual block as it is described in the paper "Spike-based residual blocks".
    See https://arxiv.org/abs/2102.04159

    TODO? Use a more optimized version, like Wei's ?
    """
    def __init__(self, in_channels: int, connect_function='ADD', use_plif=True, tau=2., v_threshold=1., v_reset=0.):
        super(cext_SEWResBlock, self).__init__()

        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.sn1 = MultiStepParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True)

        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.sn2 = MultiStepParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True)

        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.sn2(out)

        if self.connect_function == 'ADD':
            out += identity
        elif self.connect_function == 'MUL' or self.connect_f == 'AND':
            out *= identity
        elif self.connect_function == 'OR':
            out = SpikesOR.apply(out, identity)
        elif self.connect_function == 'NMUL':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class SEWResBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) residual block as it is described in the paper "Spike-based residual blocks".
    See https://arxiv.org/abs/2102.04159

    TODO? Use a more optimized version, like Wei's ?
    """
    def __init__(self, in_channels: int, connect_function='ADD', use_plif=True, tau=2., v_threshold=1., v_reset=0.):
        super(SEWResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.sn1 = ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.sn2 = ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True)

        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.sn2(out)

        if self.connect_function == 'ADD':
            out += identity
        elif self.connect_function == 'MUL' or self.connect_f == 'AND':
            out *= identity
        elif self.connect_function == 'OR':
            out = SpikesOR.apply(out, identity)
        elif self.connect_function == 'NMUL':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

