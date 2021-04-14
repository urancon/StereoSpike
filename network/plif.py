import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils import cpp_extension
from spikingjelly.cext import neuron as cext_neuron
from spikingjelly.cext.neuron import _C_neuron
use_fast_math = True
extra_cuda_cflags = []
if use_fast_math:
    extra_cuda_cflags.append('-use_fast_math')

_C_PLIF = cpp_extension.load(name='neuron_plif', sources=['network/csrc/plif.cpp', 'network/csrc/plif_kernel.cu'],
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True)


class ParametricLIFStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input):
        if v_reset is None:
            raise NotImplementedError

        spike, v_next, grad_s_to_h, grad_v_to_h, grad_h_to_rtau = _C_PLIF.ParametricLIF_hard_reset_forward_with_grad(x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h, grad_h_to_rtau)
        ctx.reciprocal_tau = reciprocal_tau
        ctx.detach_input = detach_input

        return spike, v_next

    @staticmethod
    def backward(ctx, grad_spike, grad_v_next):
        grad_x, grad_v, grad_rtau = _C_PLIF.ParametricLIF_backward(grad_spike, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.saved_tensors[2], ctx.reciprocal_tau, ctx.detach_input)
        return grad_x, grad_v, None, None, None, None, None, grad_rtau, None


class ParametricLIFMultiStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input):
        if v_reset is None:
            raise NotImplementedError

        spike_seq, v_next, grad_s_to_h, grad_v_to_h, grad_h_to_rtau = _C_PLIF.ParametricLIF_hard_reset_fptt_with_grad(x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h, grad_h_to_rtau)
        ctx.reciprocal_tau = reciprocal_tau
        ctx.detach_input = detach_input

        return spike_seq, v_next

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_next):
        grad_x_seq, grad_v, grad_rtau = _C_PLIF.ParametricLIF_bptt(grad_spike_seq, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.saved_tensors[2], ctx.reciprocal_tau, ctx.detach_input)
        return grad_x_seq, grad_v, None, None, None, None, None, grad_rtau, None


class ParametricLIFNode(cext_neuron.BaseNode):
    def __init__(self, init_tau=2.0, detach_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0,
                 detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset)
        self.w = nn.Parameter(torch.Tensor([- math.log(init_tau - 1)]))
        self.detach_input = detach_input

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike, self.v = ParametricLIFStep.apply(dv, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset,
                                              self.grad_surrogate_function_index, self.w.sigmoid(), self.detach_input)
            else:
                spike, self.v = _C_neuron.LIF_hard_reset_forward(dv, self.v, self.v_threshold, self.v_reset, self.w.sigmoid(), self.detach_input)
            return spike


class MultiStepParametricLIFNode(ParametricLIFNode):

    def forward(self, dv_seq: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv_seq[0].data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike_seq, self.v = ParametricLIFMultiStep.apply(dv_seq, self.v, self.v_threshold, self.v_reset, self.alpha,
                                                       self.detach_reset, self.grad_surrogate_function_index, self.w.sigmoid(), self.detach_input)
            else:
                spike_seq, self.v = _C_neuron.LIF_hard_reset_fptt(dv_seq, self.v, self.v_threshold, self.v_reset, self.w.sigmoid(), self.detach_input)
            return spike_seq


class StatelessMultiStepParametricLIFNode(ParametricLIFNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0,
                 detach_reset=False):
        super().__init__(tau, v_threshold, v_reset, surrogate_function, alpha, detach_reset)
        del self.v

    def forward(self, dv_seq: torch.Tensor, v_init=None, return_v_next=False):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if v_init is None:
                v_init = torch.zeros_like(dv_seq[0].data)
            if self.training:
                spike_seq, v_next = ParametricLIFMultiStep.apply(dv_seq, v_init, self.v_threshold, self.v_reset, self.alpha,
                                                       self.detach_reset, self.grad_surrogate_function_index, self.w.sigmoid())
            else:
                spike_seq, v_next = _C_neuron.LIF_hard_reset_fptt(dv_seq, v_init, self.v_threshold, self.v_reset, self.w.sigmoid())
            if return_v_next:
                return spike_seq, v_next
            else:
                return spike_seq