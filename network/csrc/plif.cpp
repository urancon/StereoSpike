#include <iostream>
#include <torch/extension.h>

std::vector<at::Tensor> ParametricLIF_hard_reset_forward_with_grad(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
    const float & reciprocal_tau, const bool & detach_x);

std::vector<at::Tensor> ParametricLIF_hard_reset_fptt_with_grad(
    torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset, 
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
    const float & reciprocal_tau, const bool & detach_x);

std::vector<at::Tensor> ParametricLIF_backward(
    torch::Tensor & grad_spike, torch::Tensor & grad_v_next, torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h, torch::Tensor & grad_h_to_rtau,
    const float & reciprocal_tau, const bool & detach_x);

std::vector<at::Tensor> ParametricLIF_bptt(
    torch::Tensor & grad_spike_seq, torch::Tensor & grad_v_next,
    torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h, torch::Tensor & grad_h_to_rtau,
    const float & reciprocal_tau, const bool & detach_x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ParametricLIF_hard_reset_forward_with_grad", &ParametricLIF_hard_reset_forward_with_grad);
    m.def("ParametricLIF_backward", &ParametricLIF_backward);
    m.def("ParametricLIF_hard_reset_fptt_with_grad", &ParametricLIF_hard_reset_fptt_with_grad);
    m.def("ParametricLIF_bptt", &ParametricLIF_bptt);
}
