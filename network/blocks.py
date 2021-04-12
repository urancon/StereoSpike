import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from spikingjelly.clock_driven import functional, layer, surrogate
from spikingjelly.clock_driven import neuron


class OneToOne(nn.Module):
    """ Applies an element-wise multiplication between the input tensor and a learnable weight tensor of the same shape.
    Can be interpreted as a one-to-one connection between neurons from an input [W, H] map and an output [W, H] map.

    The need for this emerged in our problem because spikes from the last LIF layer of our networks do not "carry" the
     same depth quantity depending of their position in the field of view. Pixels at the bottom of the image will mostly
     be ground pixels for instance, and there the depth variations are inferior than those on the top of the image.

    UPDATE: it actually turns out not to be a good idea to use such weight connections between the last LIF and IF
     neurons. Proof:

        So we have LIF -> one-to-one -> IF at the end of the SNN.
        Suppose, for a particular output IF neuron at time t, a depth of 10 m to predict with its non-leaking potential.
        The preceding LIF layer could emit 10 spikes, and the one-to-one weights between the two could be of 1 m/spk.
        Therefore, our IF neuron gets a right membrane potential of 10 m. So far so good!

        Now suppose, at the next timestep t+1, a depth to predict of 9 m instead of 10 for the same IF neuron.
        So it should decrease its potential of 1 m.
        But the only way for it to do so is by receiving negative-weighted spikes from the preceding layer.
        For instance, the preceding LIF neuron would emit 1 spike with a one-to-one weight of -1 m/spk.
        But that cannot be, since this weight already equals +1 m/spk !
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


class SEWResBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) residual block as it is described in the paper "Spike-based residual blocks".
    See https://arxiv.org/abs/2102.04159

    TODO: Use Wei's code for different connect functions.
    TODO? Use a more optimized version, like Wei's ?
    """
    def __init__(self, in_channels: int, connect_function='ADD', tau=2., v_threshold=1., v_reset=0.):
        super(SEWResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(in_channels),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False),
            nn.BatchNorm2d(in_channels),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        out = self.layers(x)

        if self.connect_function == 'ADD':
            out += identity
        elif self.connect_function == 'MUL' or self.connect_f == 'AND':
            out *= identity
        elif self.connect_function == 'OR':
            raise NotImplementedError(self.connect_f)
            # out = SpikesOR.apply(out, identity)
        elif self.connect_function == 'NMUL':
            raise NotImplementedError(self.connect_f)
            # out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

