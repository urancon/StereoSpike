import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from spikingjelly.clock_driven import functional, surrogate, neuron, layer, rnn

from .plif import ParametricLIFNode
from .plif2 import LearnableVthPLIFNode
from .csrc.element_wise import SpikesOR


class AttentionGate(nn.Module):
    """
    An ANN attention gate to add to encoder-decoder architectures just like in the paper 'Attention U-Net: Learning
     Where to Look for the Pancreas'.

     g: tensor from the next lowest decoder layer
     x: tensor from the skip connection

    This constitutes a 'soft' attention mechanism.
    """
    def __init__(self, g_channels: int, x_channels: int, out_size: tuple):
        super(AttentionGate, self).__init__()

        # g keeps its smaller spatial dimension, but gets the same number of channels as x
        self.conv_g = nn.Conv2d(g_channels, x_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # x keeps its lower number of channels, but gets the same spatial dimensions as g
        self.conv_x = nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=2, padding=0, bias=False)

        # layers for post-processing of combined g and x information
        self.collapse = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            nn.UpsamplingBilinear2d(size=out_size)
        )

    def forward(self, g, x):
        convG = self.conv_g(g)
        convX = self.conv_x(x)

        combined = convG + self.crop_like(convX, convG)
        combined = self.collapse(combined)

        out = x * combined

        return out

    @staticmethod
    def crop_like(convX, convG):
        return convX[:, :, :convG.size(2), :convG.size(3)]


class BilinConvUpsampling(nn.Module):
    """
    Upsampling block made to address the production of checkerboard artifacts by transposed convolutions.
    Nearest neighbour (NN) upsampling combined to regular convolutions efficiently gets rid of these patterns.
    Linear interpolation (among others) can produce non-integer values (e.g. 0.5 between 0 and 1), which go against the
    philosophy of spiking neural networks, which receive integer amounts of input spikes. Having noticed that, NN
    upsampling is compatible with SNN's philosophy and could be implemented on dedicated hardware.

    See https://distill.pub/2016/deconv-checkerboard/ for more insights on this phenomenon.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias: bool = False):
        super(BilinConvUpsampling, self).__init__()

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        out = self.up(x)
        return out


class ResBlock(nn.Module):
    """
    ANN
    """
    def __init__(self, in_channels: int, connect_function='ADD', bias: bool = False):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3-1)//2, bias=bias),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(in_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3-1)//2, bias=bias),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(in_channels),
        )

        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

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


class ConvLSTMCell(nn.Module):
    """ANN """

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        self.Gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=pad)

        self.h = None
        self.c = None

    def forward(self, x):

        if self.h is None:
            self.h = torch.zeros(size=[x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]], dtype=torch.float,
                                 device=x.device)
        if self.c is None:
            self.c = torch.zeros(size=[x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]], dtype=torch.float,
                                 device=x.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((x, self.h), dim=1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        self.c = (remember_gate * self.c) + (in_gate * cell_gate)
        self.h = out_gate * torch.tanh(self.c)

        return self.h

    def set_hc(self, h, c):
        self.h = h
        self.c = c


class SpikingConvLSTMCell(rnn.SpikingRNNCellBase):
    """
    Inspired from ConvLSTM and the paper 'Long Short-Term Memory Spiking Networks and Their Applications'.
    The use of a network with this module is almost transparent to one without.
    
    input vector x: (B, C_in, H, W)
    hidden states h and c: (B, C_hidd, H, W)
    """
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int, bias: bool = False,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        super().__init__(input_channels, hidden_channels, bias)

        self.hidden_channels = hidden_channels

        self.conv_ih = nn.Conv2d(input_channels, 4 * hidden_channels, kernel_size=kernel_size, padding=kernel_size//2,
                                 bias=bias)
        self.conv_hh = nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size=kernel_size, padding=kernel_size//2,
                                 bias=bias)

        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.h = None
        self.c = None

        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        if self.h is None:
            self.h = torch.zeros(size=[x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]], dtype=torch.float,
                            device=x.device)
        if self.c is None:
            self.c = torch.zeros(size=[x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]], dtype=torch.float,
                            device=x.device)

        if self.surrogate_function2 is None:
            i, f, g, o = torch.split(self.surrogate_function1(self.conv_ih(x) + self.conv_hh(self.h)),
                                     self.hidden_channels, dim=1)
        else:
            i, f, g, o = torch.split(self.conv_ih(x) + self.conv_hh(self.h), self.hidden_channels, dim=1)
            i = self.surrogate_function1(i)
            f = self.surrogate_function1(f)
            g = self.surrogate_function2(g)
            o = self.surrogate_function1(o)

        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.c = self.c * f + i * g
        self.h = self.c * o

        return self.h

    def set_hc(self, h, c):
        self.h = h
        self.c = c


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
        return torch.sum(torch.mul(input, self.weight), axis=1).unsqueeze(dim=1)  # axis=1 because we sum along channel dimension

    def extra_repr(self) -> str:
        return 'in_features={}'.format(
            self.in_features is not None
        )


class MultiplyBy(nn.Module):
    """
    Could potentially replace BatchNorm2d to get rid of the offset induced by it.
    By multiplying input values by a certain parameter, it should allow subsequent PLIFNodes to actually spike and solve
     the vanishing spike phenomenon that is observed without BatchNorm.

    TODO: instead of learning it, make the scale_value gradually decay as we go further into the sequence ?
    """

    def __init__(self, scale_value: float = 5., learnable: bool = False) -> None:
        super(MultiplyBy, self).__init__()

        if learnable:
            self.scale_value = Parameter(Tensor([scale_value]))
        else:
            self.scale_value = scale_value

    def forward(self, input: Tensor) -> Tensor:
        return torch.mul(input, self.scale_value)


class NNConvUpsampling(nn.Module):
    """
    Upsampling block made to address the production of checkerboard artifacts by transposed convolutions.
    Nearest neighbour (NN) upsampling combined to regular convolutions efficiently gets rid of these patterns.
    Linear interpolation (among others) can produce non-integer values (e.g. 0.5 between 0 and 1), which go against the
    philosophy of spiking neural networks, which receive integer amounts of input spikes. Having noticed that, NN
    upsampling is compatible with SNN's philosophy and could be implemented on dedicated hardware.

    See https://distill.pub/2016/deconv-checkerboard/ for more insights on this phenomenon.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias: bool = False):
        super(NNConvUpsampling, self).__init__()

        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        out = self.up(x)
        return out


class SEWResBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) residual block as it is described in the paper "Spike-based residual blocks".
    See https://arxiv.org/abs/2102.04159
    """
    def __init__(self, in_channels: int, connect_function='ADD', use_plif=True, tau=2., v_threshold=1., v_reset=0.):
        super(SEWResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            MultiplyBy(),
            #nn.BatchNorm2d(in_channels),
            #nn.BatchNorm2d(in_channels, affine=False),
        )

        self.sn1 = ParametricLIFNode(init_tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_input=True, detach_reset=True) if use_plif else neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=True)

        self.conv2 = nn.Sequential(
            #layer.Dropout2d(p=0.2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            MultiplyBy(),
            #nn.BatchNorm2d(in_channels),
            #nn.BatchNorm2d(in_channels, affine=False),
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
