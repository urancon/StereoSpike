#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>

__global__ void spikes_or_cuda_kernel(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ z, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
      z[index] = __saturatef(x[index] + y[index]);
  }
}

__global__ void spikes_or_cuda_kernel_half(const at::Half* __restrict__ x, const at::Half* __restrict__ y, at::Half* __restrict__ z, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
      z[index] = __hadd_sat(x[index], y[index]);
  }
}

torch::Tensor spikes_or(torch::Tensor & x, torch::Tensor & y)
{
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    if (! x.is_contiguous())
    {
        x = x.contiguous();
    }
    if (! y.is_contiguous())
    {
        y = y.contiguous();
    }
    auto z = torch::zeros_like(x.data());
    const int size = x.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    cudaSetDevice(x.get_device());
    if (x.scalar_type() == c10::ScalarType::Float)
    {
      spikes_or_cuda_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), size);
    }
    else if (x.scalar_type() == c10::ScalarType::Half)
    {
      spikes_or_cuda_kernel_half<<<blocks, threads>>>(x.data_ptr<at::Half>(), y.data_ptr<at::Half>(), z.data_ptr<at::Half>(), size);
    }
    return z;   
}

