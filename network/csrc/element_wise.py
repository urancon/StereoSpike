import torch
# from torch.utils import cpp_extension
# use_fast_math = True
# extra_cuda_cflags = []
# if use_fast_math:
#     extra_cuda_cflags.append('-use_fast_math')
#
# _C_element_wise = cpp_extension.load(name='element_wise', sources=['./csrc/element_wise.cpp', './csrc/element_wise_kernel.cu'],
#     extra_cuda_cflags=extra_cuda_cflags,
#     verbose=True)

# class SpikesOR(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, y):
#         return _C_element_wise.spikes_or(x, y)
#
#     @staticmethod
#     def backward(ctx, grad_z):
#         return grad_z, grad_z


class SpikesOR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return torch.clamp_max(x + y, 1.0)

    @staticmethod
    def backward(ctx, grad_z):
        return grad_z, grad_z

class SpikesXOR(torch.autograd.Function):
    # a⊕b = (¬a ∧ b) ∨ (a ∧¬b)
    # ((1 - a) * b) or (a * (1 - b))
    @staticmethod
    def forward(ctx, x, y):
        if x.requires_grad or y.requires_grad:
            ctx.save_for_backward(x, y)
        return torch.logical_xor(x.bool(), y.bool()).to(x)

    @staticmethod
    def backward(ctx, grad_z):
        x = ctx.saved_tensors[0]
        y = ctx.saved_tensors[1]
        return grad_z * (1.0 - 2 * y), grad_z * (1.0 - 2 * x)