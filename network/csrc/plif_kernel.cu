#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <math.h>
#include <stdio.h>
#define CHECK_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x" must be a CUDA tensor");if (! x.is_contiguous()){x = x.contiguous();}
#define CHECK_CUDA_OPERATION(operation) if(operation != cudaSuccess){printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));exit(-1);}
#define THREADS 1024

__forceinline__  __device__ float grad_atan(const float & alpha, const float & x)
{
  const float M_PI_2__alpha__x = (float) M_PI_2 * alpha * x;
  return alpha / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
}

__forceinline__  __device__ float grad_sigmoid(const float & alpha, const float & x)
{
  const float sigmoid_ax = 1.0f / (1.0f + expf(- alpha * x));
  return (1.0f - sigmoid_ax) * sigmoid_ax * alpha;
}

typedef float (*grad_surrogate_function) (const float &, const float &);

__device__ const grad_surrogate_function grad_surrogate_function_pointer[2] = { 
    grad_atan, 
    grad_sigmoid
    };


__forceinline__  __device__ half grad_atan_half(const half & alpha, const half & x)
{
  #if __CUDACC_VER_MAJOR__ >= 11
  const half M_PI_2__alpha__x = __hmul(__hmul(__double2half(M_PI_2), alpha), x);
  #else
  const half M_PI_2__alpha__x = __hmul(__hmul(__float2half((float) M_PI_2), alpha), x);
  #endif
  return __hdiv(__hdiv(alpha, __float2half(2.0f)), __hfma(M_PI_2__alpha__x, M_PI_2__alpha__x, __float2half(1.0f)));
}

__forceinline__  __device__ half grad_sigmoid_half(const half & alpha, const half & x)
{
  const half sigmoid_ax = __hdiv(__float2half(1.0f), __hadd(hexp(__hneg(__hmul(alpha, x))), __float2half(1.0f)));
  return __hmul(__hmul(__hsub(__float2half(1.0f), sigmoid_ax), sigmoid_ax), alpha);
}

typedef half (*grad_surrogate_function_half) (const half &, const half &);

__device__ const grad_surrogate_function_half grad_surrogate_function_pointer_half[2] = { 
    grad_atan_half, 
    grad_sigmoid_half
    };
    
__global__ void ParametricLIF_hard_reset_forward_with_grad_cuda_kernel(
    const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next,
    float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h, float* __restrict__ grad_h_to_rtau,
    const float v_th, const float v_reset, const int size,
    const float alpha, const bool detach_reset, const int grad_surrogate_function_index,
    const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    grad_h_to_rtau[index] = x[index] - v[index] + v_reset;
    const float h = v[index] + reciprocal_tau * grad_h_to_rtau[index];
    if (h >= v_th)
    {
      spike[index] = 1.0f;
      v_next[index] = v_reset;
    }
    else
    {
      spike[index] = 0.0f;
      v_next[index] = h;
    }
    grad_s_to_h[index] = grad_surrogate_function_pointer[grad_surrogate_function_index](alpha, h - v_th);
    grad_v_to_h[index] = 1.0f - spike[index] + (v_reset - h) * grad_s_to_h[index] * (1.0f - (float) detach_reset);
  }
}

__global__ void ParametricLIF_hard_reset_forward_with_grad_cuda_kernel_half(
  const at::Half* __restrict__ x, const at::Half* __restrict__ v, at::Half* __restrict__ spike, at::Half* __restrict__ v_next,
  at::Half* __restrict__ grad_s_to_h, at::Half* __restrict__ grad_v_to_h, at::Half* __restrict__ grad_h_to_rtau,
  const half v_th, const half v_reset, const int size,
  const half alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const half reciprocal_tau)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size)
{
  grad_h_to_rtau[index] = __hadd(__hsub(x[index], v[index]), v_reset);
  const half h = __hfma(reciprocal_tau, grad_h_to_rtau[index], v[index]);
  if (__hgeu(h, v_th))
  {
    spike[index] = __float2half(1.0f);
    v_next[index] = v_reset;
  }
  else
  {
    spike[index] = __float2half(0.0f);
    v_next[index] = h;
  }
  grad_s_to_h[index] = grad_surrogate_function_pointer_half[grad_surrogate_function_index](alpha, __hsub(h, v_th));
  grad_v_to_h[index] = __hfma(__hmul(__hsub(v_reset, h), grad_s_to_h[index]), __float2half(1.0f - (float) detach_reset), __hsub(__float2half(1.0f), spike[index]));
}
}
//PLIF detach x
__global__ void ParametricLIF_detach_x_hard_reset_forward_with_grad_cuda_kernel(
  const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next,
  float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h, float* __restrict__ grad_h_to_rtau,
  const float v_th, const float v_reset, const int size,
  const float alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    grad_h_to_rtau[index] = v_reset - v[index];
    const float h = v[index] + x[index] + reciprocal_tau * grad_h_to_rtau[index];
    if (h >= v_th)
    {
      spike[index] = 1.0f;
      v_next[index] = v_reset;
    }
    else
    {
      spike[index] = 0.0f;
      v_next[index] = h;
    }
    grad_s_to_h[index] = grad_surrogate_function_pointer[grad_surrogate_function_index](alpha, h - v_th);
    grad_v_to_h[index] = 1.0f - spike[index] + (v_reset - h) * grad_s_to_h[index] * (1.0f - (float) detach_reset);
  }
}

__global__ void ParametricLIF_detach_x_hard_reset_forward_with_grad_cuda_kernel_half(
const at::Half* __restrict__ x, const at::Half* __restrict__ v, at::Half* __restrict__ spike, at::Half* __restrict__ v_next,
at::Half* __restrict__ grad_s_to_h, at::Half* __restrict__ grad_v_to_h, at::Half* __restrict__ grad_h_to_rtau,
const half v_th, const half v_reset, const int size,
const half alpha, const bool detach_reset, const int grad_surrogate_function_index,
const half reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    grad_h_to_rtau[index] = __hsub(v_reset, v[index]);
    const half h = __hfma(reciprocal_tau, grad_h_to_rtau[index], __hadd((half) v[index], (half) x[index]));
    if (__hgeu(h, v_th))
    {
      spike[index] = __float2half(1.0f);
      v_next[index] = v_reset;
    }
    else
    {
      spike[index] = __float2half(0.0f);
      v_next[index] = h;
    }
    grad_s_to_h[index] = grad_surrogate_function_pointer_half[grad_surrogate_function_index](alpha, __hsub(h, v_th));
    grad_v_to_h[index] = __hfma(__hmul(__hsub(v_reset, h), grad_s_to_h[index]), __float2half(1.0f - (float) detach_reset), __hsub(__float2half(1.0f), spike[index]));
  }
} 

std::vector<at::Tensor> ParametricLIF_hard_reset_forward_with_grad(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset,
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
  const float & reciprocal_tau, const bool & detach_x)
{   
    CHECK_TENSOR(x);
    CHECK_TENSOR(v);

    auto spike = torch::zeros_like(v.data());
    auto v_next = spike.data().clone();
    auto grad_s_to_h = spike.data().clone();
    auto grad_v_to_h = spike.data().clone();
    auto grad_h_to_rtau = spike.data().clone();

    CHECK_TENSOR(spike);
    CHECK_TENSOR(v_next);
    CHECK_TENSOR(grad_s_to_h);
    CHECK_TENSOR(grad_v_to_h);
    CHECK_TENSOR(grad_h_to_rtau);

    const int size = x.numel();
    const int threads = THREADS;
    const int blocks = (size + threads - 1) / threads;
    CHECK_CUDA_OPERATION(cudaSetDevice(x.get_device()));
    if (x.scalar_type() == c10::ScalarType::Float)
    {
      if (detach_x)
      {
        ParametricLIF_detach_x_hard_reset_forward_with_grad_cuda_kernel<<<blocks, threads>>>(
          x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(), 
          grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
          v_th, v_reset, size, 
          alpha, detach_reset, grad_surrogate_function_index,
          reciprocal_tau);
      }
      else
      {

        ParametricLIF_hard_reset_forward_with_grad_cuda_kernel<<<blocks, threads>>>(
          x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(), 
          grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
          v_th, v_reset, size, 
          alpha, detach_reset, grad_surrogate_function_index,
          reciprocal_tau);
      }
      
    }
    else if (x.scalar_type() == c10::ScalarType::Half)
    {
      if (detach_x)
      {
        ParametricLIF_detach_x_hard_reset_forward_with_grad_cuda_kernel_half<<<blocks, threads>>>(
          x.data_ptr<at::Half>(), v.data_ptr<at::Half>(), spike.data_ptr<at::Half>(), v_next.data_ptr<at::Half>(), 
          grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
          __float2half(v_th), __float2half(v_reset), size, 
          __float2half(alpha), detach_reset, grad_surrogate_function_index,
          __float2half(reciprocal_tau));
      }
      else
      {
        ParametricLIF_hard_reset_forward_with_grad_cuda_kernel_half<<<blocks, threads>>>(
          x.data_ptr<at::Half>(), v.data_ptr<at::Half>(), spike.data_ptr<at::Half>(), v_next.data_ptr<at::Half>(), 
          grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
          __float2half(v_th), __float2half(v_reset), size, 
          __float2half(alpha), detach_reset, grad_surrogate_function_index,
          __float2half(reciprocal_tau));
      }

    }
    return {spike, v_next, grad_s_to_h, grad_v_to_h, grad_h_to_rtau};
}

__global__ void ParametricLIF_hard_reset_fptt_with_grad_cuda_kernel(
  const float* __restrict__ x_seq, float* __restrict__ spike_seq, float* __restrict__ v_next, 
  float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h, float* __restrict__ grad_h_to_rtau, 
  const float v_th, const float v_reset, const int neuron_num, const int size,
  const float alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h_to_rtau[mem_index] = x_seq[mem_index] - v_next[index] + v_reset;
      const float h = v_next[index] + reciprocal_tau * grad_h_to_rtau[mem_index];
      if (h >= v_th)
      {
        spike_seq[mem_index] = 1.0f;
        v_next[index] = v_reset;
      }
      else
      {
        spike_seq[mem_index] = 0.0f;
        v_next[index] = h;
      }
      grad_s_to_h[mem_index] = grad_surrogate_function_pointer[grad_surrogate_function_index](alpha, h - v_th);
      grad_v_to_h[mem_index] = 1.0f - spike_seq[mem_index] + (v_reset - h) * grad_s_to_h[mem_index] * (1.0f - (float) detach_reset);
    }
    
  }
}

__global__ void ParametricLIF_hard_reset_fptt_with_grad_cuda_kernel_half(
  const at::Half* __restrict__ x_seq, at::Half* __restrict__ spike_seq, at::Half* __restrict__ v_next, 
  at::Half* __restrict__ grad_s_to_h, at::Half* __restrict__ grad_v_to_h, at::Half* __restrict__ grad_h_to_rtau, 
  const half v_th, const half v_reset, const int neuron_num, const int size,
  const half alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const half reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h_to_rtau[mem_index] = __hadd(__hsub(x_seq[mem_index], v_next[index]), v_reset);
      const half h = __hfma(reciprocal_tau, grad_h_to_rtau[mem_index], v_next[index]);
      if (__hgeu(h, v_th))
      {
        spike_seq[mem_index] = __float2half(1.0f);
        v_next[index] = v_reset;
      }
      else
      {
        spike_seq[mem_index] = __float2half(0.0f);
        v_next[index] = h;
      }
      grad_s_to_h[mem_index] = grad_surrogate_function_pointer_half[grad_surrogate_function_index](alpha, __hsub(h, v_th));
      grad_v_to_h[mem_index] = __hfma(__hmul(__hsub(v_reset, h), grad_s_to_h[mem_index]), __float2half(1.0f - (float) detach_reset), __hsub(__float2half(1.0f), spike_seq[mem_index]));
    }
    
  }
}

//PLIF detach x

__global__ void ParametricLIF_detach_x_hard_reset_fptt_with_grad_cuda_kernel(
  const float* __restrict__ x_seq, float* __restrict__ spike_seq, float* __restrict__ v_next, 
  float* __restrict__ grad_s_to_h, float* __restrict__ grad_v_to_h, float* __restrict__ grad_h_to_rtau, 
  const float v_th, const float v_reset, const int neuron_num, const int size,
  const float alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const float reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h_to_rtau[mem_index] = v_reset - v_next[index];
      const float h = v_next[index] + x_seq[mem_index] + reciprocal_tau * grad_h_to_rtau[mem_index];

      if (h >= v_th)
      {
        spike_seq[mem_index] = 1.0f;
        v_next[index] = v_reset;
      }
      else
      {
        spike_seq[mem_index] = 0.0f;
        v_next[index] = h;
      }
      grad_s_to_h[mem_index] = grad_surrogate_function_pointer[grad_surrogate_function_index](alpha, h - v_th);
      grad_v_to_h[mem_index] = 1.0f - spike_seq[mem_index] + (v_reset - h) * grad_s_to_h[mem_index] * (1.0f - (float) detach_reset);
    }
    
  }
}

__global__ void ParametricLIF_detach_x_hard_reset_fptt_with_grad_cuda_kernel_half(
  const at::Half* __restrict__ x_seq, at::Half* __restrict__ spike_seq, at::Half* __restrict__ v_next, 
  at::Half* __restrict__ grad_s_to_h, at::Half* __restrict__ grad_v_to_h, at::Half* __restrict__ grad_h_to_rtau, 
  const half v_th, const half v_reset, const int neuron_num, const int size,
  const half alpha, const bool detach_reset, const int grad_surrogate_function_index,
  const half reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    for(int mem_offset = 0; mem_offset < size; mem_offset += neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h_to_rtau[mem_index] = __hsub(v_reset, v_next[index]);
      const half h = __hfma(reciprocal_tau, grad_h_to_rtau[mem_index], __hadd((half) v_next[index], (half) x_seq[mem_index]));

      if (__hgeu(h, v_th))
      {
        spike_seq[mem_index] = __float2half(1.0f);
        v_next[index] = v_reset;
      }
      else
      {
        spike_seq[mem_index] = __float2half(0.0f);
        v_next[index] = h;
      }
      grad_s_to_h[mem_index] = grad_surrogate_function_pointer_half[grad_surrogate_function_index](alpha, __hsub(h, v_th));
      grad_v_to_h[mem_index] = __hfma(__hmul(__hsub(v_reset, h), grad_s_to_h[mem_index]), __float2half(1.0f - (float) detach_reset), __hsub(__float2half(1.0f), spike_seq[mem_index]));
    }
    
  }
}


std::vector<at::Tensor> ParametricLIF_hard_reset_fptt_with_grad(
  torch::Tensor & x_seq, torch::Tensor & v, const float & v_th, const float & v_reset, 
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
  const float & reciprocal_tau, const bool & detach_x)
{
    CHECK_TENSOR(x_seq);
    CHECK_TENSOR(v);
    auto spike_seq = torch::zeros_like(x_seq.data());
    auto v_next = v.data().clone();
    auto grad_s_to_h = spike_seq.data().clone();
    auto grad_v_to_h = spike_seq.data().clone();
    auto grad_h_to_rtau = spike_seq.data().clone();
    CHECK_TENSOR(spike_seq);
    CHECK_TENSOR(v_next);
    CHECK_TENSOR(grad_s_to_h);
    CHECK_TENSOR(grad_v_to_h);
    CHECK_TENSOR(grad_h_to_rtau);

    const int seq_len = x_seq.size(0);
    const int size = x_seq.numel();
    const int threads = THREADS;
    const int neuron_num = size / seq_len;
    const int blocks = (neuron_num + threads - 1) / threads;
    CHECK_CUDA_OPERATION(cudaSetDevice(x_seq.get_device()));
    if (x_seq.scalar_type() == c10::ScalarType::Float)
    {
      if (detach_x)
      {
        ParametricLIF_detach_x_hard_reset_fptt_with_grad_cuda_kernel<<<blocks, threads>>>(
          x_seq.data_ptr<float>(), spike_seq.data_ptr<float>(), v_next.data_ptr<float>(), 
          grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
          v_th, v_reset, neuron_num, size, 
          alpha, detach_reset, grad_surrogate_function_index,
          reciprocal_tau);
      }
      else
      {
        ParametricLIF_hard_reset_fptt_with_grad_cuda_kernel<<<blocks, threads>>>(
          x_seq.data_ptr<float>(), spike_seq.data_ptr<float>(), v_next.data_ptr<float>(), 
          grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
          v_th, v_reset, neuron_num, size, 
          alpha, detach_reset, grad_surrogate_function_index,
          reciprocal_tau);
      }

    }
    else if (x_seq.scalar_type() == c10::ScalarType::Half)
    {
      if (detach_x)
      {
        ParametricLIF_detach_x_hard_reset_fptt_with_grad_cuda_kernel_half<<<blocks, threads>>>(
          x_seq.data_ptr<at::Half>(), spike_seq.data_ptr<at::Half>(), v_next.data_ptr<at::Half>(), 
          grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
          __float2half(v_th), __float2half(v_reset), neuron_num, size, 
          __float2half(alpha), detach_reset, grad_surrogate_function_index,
          __float2half(reciprocal_tau));
      }
      else
      {
        ParametricLIF_hard_reset_fptt_with_grad_cuda_kernel_half<<<blocks, threads>>>(
          x_seq.data_ptr<at::Half>(), spike_seq.data_ptr<at::Half>(), v_next.data_ptr<at::Half>(), 
          grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
          __float2half(v_th), __float2half(v_reset), neuron_num, size, 
          __float2half(alpha), detach_reset, grad_surrogate_function_index,
          __float2half(reciprocal_tau));
      }

    }
    return {spike_seq, v_next, grad_s_to_h, grad_v_to_h, grad_h_to_rtau};
}

__global__ void ParametricLIF_backward_cuda_kernel(
  float* __restrict__ grad_x, float* __restrict__ grad_v, float* __restrict__ grad_rtau,
  const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next, 
  const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h, const float* __restrict__ grad_h_to_rtau,
  const int size,
  const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdata[THREADS];
  if (index < size)
  {
    const float grad_h = grad_spike[index] * grad_s_to_h[index] + grad_v_next[index] * grad_v_to_h[index];
    grad_x[index] = grad_h * reciprocal_tau;
    grad_v[index] = grad_h * one_sub_reciprocal_tau;
    sdata[threadIdx.x] = grad_h * grad_h_to_rtau[index];
  }
  else
  {
    sdata[threadIdx.x] = 0.0f;
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}

__global__ void ParametricLIF_backward_cuda_kernel_half(
  at::Half* __restrict__ grad_x, at::Half* __restrict__ grad_v, at::Half* __restrict__ grad_rtau,
  const at::Half* __restrict__ grad_spike, const at::Half* __restrict__ grad_v_next, 
  const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h, const at::Half* __restrict__ grad_h_to_rtau,
  const int size,
  const half reciprocal_tau, const half one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ half sdata[THREADS];
  if (index < size)
  {
    const half grad_h = __hfma(grad_spike[index], grad_s_to_h[index], __hmul(grad_v_next[index], grad_v_to_h[index]));
    grad_x[index] = __hmul(grad_h, reciprocal_tau);
    grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
    sdata[threadIdx.x] = __hmul(grad_h, grad_h_to_rtau[index]);
  }
  else
  {
    sdata[threadIdx.x] = __float2half(0.0f);
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] = __hadd(sdata[threadIdx.x + stride], sdata[threadIdx.x]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}

//detach x-------
__global__ void ParametricLIF_detach_x_backward_cuda_kernel(
  float* __restrict__ grad_x, float* __restrict__ grad_v, float* __restrict__ grad_rtau,
  const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next, 
  const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h, const float* __restrict__ grad_h_to_rtau,
  const int size,
  const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdata[THREADS];
  if (index < size)
  {
    const float grad_h = grad_spike[index] * grad_s_to_h[index] + grad_v_next[index] * grad_v_to_h[index];
    grad_x[index] = grad_h;
    grad_v[index] = grad_h * one_sub_reciprocal_tau;
    sdata[threadIdx.x] = grad_h * grad_h_to_rtau[index];
  }
  else
  {
    sdata[threadIdx.x] = 0.0f;
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}

__global__ void ParametricLIF_detach_x_backward_cuda_kernel_half(
  at::Half* __restrict__ grad_x, at::Half* __restrict__ grad_v, at::Half* __restrict__ grad_rtau,
  const at::Half* __restrict__ grad_spike, const at::Half* __restrict__ grad_v_next, 
  const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h, const at::Half* __restrict__ grad_h_to_rtau,
  const int size,
  const half one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ half sdata[THREADS];
  if (index < size)
  {
    const half grad_h = __hfma(grad_spike[index], grad_s_to_h[index], __hmul(grad_v_next[index], grad_v_to_h[index]));
    grad_x[index] = grad_h;
    grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
    sdata[threadIdx.x] = __hmul(grad_h, grad_h_to_rtau[index]);
  }
  else
  {
    sdata[threadIdx.x] = __float2half(0.0f);
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] = __hadd(sdata[threadIdx.x + stride], sdata[threadIdx.x]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}
std::vector<at::Tensor> ParametricLIF_backward(
  torch::Tensor & grad_spike, torch::Tensor & grad_v_next, torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h, torch::Tensor & grad_h_to_rtau,
  const float & reciprocal_tau, const bool & detach_x)
{
  CHECK_TENSOR(grad_spike);
  CHECK_TENSOR(grad_v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  CHECK_TENSOR(grad_h_to_rtau);
  auto grad_x = torch::zeros_like(grad_spike.data());
  auto grad_v = grad_x.data().clone();
  auto grad_rtau = torch::zeros({1}).to(grad_x);
  CHECK_TENSOR(grad_x);
  CHECK_TENSOR(grad_v);
  CHECK_TENSOR(grad_rtau);
  
  const int size = grad_spike.numel();
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(grad_spike.get_device()));
  if (grad_spike.scalar_type() == c10::ScalarType::Float)
  {
    if (detach_x)
    {
      ParametricLIF_detach_x_backward_cuda_kernel<<<blocks, threads>>>(
        grad_x.data_ptr<float>(), grad_v.data_ptr<float>(), grad_rtau.data_ptr<float>(), 
        grad_spike.data_ptr<float>(), grad_v_next.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
        size, 
        1.0f - reciprocal_tau);
    }
    else
    {
      ParametricLIF_backward_cuda_kernel<<<blocks, threads>>>(
        grad_x.data_ptr<float>(), grad_v.data_ptr<float>(), grad_rtau.data_ptr<float>(), 
        grad_spike.data_ptr<float>(), grad_v_next.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
        size, 
        reciprocal_tau, 1.0f - reciprocal_tau);
    }

  }
  else if (grad_spike.scalar_type() == c10::ScalarType::Half)
  {
    if (detach_x)
    {
      ParametricLIF_detach_x_backward_cuda_kernel_half<<<blocks, threads>>>(
        grad_x.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(), grad_rtau.data_ptr<at::Half>(), 
        grad_spike.data_ptr<at::Half>(), grad_v_next.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
        size, 
        __float2half(1.0f - reciprocal_tau));
    }
    else
    {
      ParametricLIF_backward_cuda_kernel_half<<<blocks, threads>>>(
        grad_x.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(), grad_rtau.data_ptr<at::Half>(), 
        grad_spike.data_ptr<at::Half>(), grad_v_next.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
        size, 
        __float2half(reciprocal_tau), __float2half(1.0f - reciprocal_tau));
    }

  }

  return {grad_x, grad_v, grad_rtau};
}

__global__ void ParametricLIF_bptt_cuda_kernel(
  float* __restrict__ grad_x_seq, float* __restrict__ grad_v, float* __restrict__ grad_rtau,
  const float* __restrict__ grad_spike_seq, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h, const float* __restrict__ grad_h_to_rtau,
  const int neuron_num, const int size,
  const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdata[THREADS];
  if (index < neuron_num)
  {
    float grad_h;
    float sum_t_grad_h_to_rtau = 0.0f;
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = grad_spike_seq[mem_index] * grad_s_to_h[mem_index] + grad_v[index] * grad_v_to_h[mem_index];
      grad_x_seq[mem_index] = grad_h * reciprocal_tau;
      grad_v[index] = grad_h * one_sub_reciprocal_tau;
      sum_t_grad_h_to_rtau += grad_h * grad_h_to_rtau[mem_index];
    }
    sdata[threadIdx.x] = sum_t_grad_h_to_rtau;
  }
  else
  {
    sdata[threadIdx.x] = 0.0f;
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}

__global__ void ParametricLIF_bptt_cuda_kernel_half(
  at::Half* __restrict__ grad_x_seq, at::Half* __restrict__ grad_v, at::Half* __restrict__ grad_rtau,
  const at::Half* __restrict__ grad_spike_seq, const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h, const at::Half* __restrict__ grad_h_to_rtau,
  const int neuron_num, const int size,
  const half reciprocal_tau, const half one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ half sdata[THREADS];
  if (index < neuron_num)
  {
    half grad_h;
    half sum_t_grad_h_to_rtau = __float2half(0.0f);
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = __hfma(grad_spike_seq[mem_index], grad_s_to_h[mem_index], __hmul(grad_v[index], grad_v_to_h[mem_index]));
      grad_x_seq[mem_index] = __hmul(grad_h, reciprocal_tau);
      grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
      sum_t_grad_h_to_rtau = __hfma(grad_h, grad_h_to_rtau[mem_index], sum_t_grad_h_to_rtau);
    }
    sdata[threadIdx.x] = sum_t_grad_h_to_rtau;
  }
  else
  {
    sdata[threadIdx.x] = __float2half(0.0f);
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] = __hadd(sdata[threadIdx.x + stride], sdata[threadIdx.x]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}

//detach x------
__global__ void ParametricLIF_detach_x_bptt_cuda_kernel(
  float* __restrict__ grad_x_seq, float* __restrict__ grad_v, float* __restrict__ grad_rtau,
  const float* __restrict__ grad_spike_seq, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h, const float* __restrict__ grad_h_to_rtau,
  const int neuron_num, const int size,
  const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdata[THREADS];
  if (index < neuron_num)
  {
    float grad_h;
    float sum_t_grad_h_to_rtau = 0.0f;
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = grad_spike_seq[mem_index] * grad_s_to_h[mem_index] + grad_v[index] * grad_v_to_h[mem_index];
      grad_x_seq[mem_index] = grad_h;
      grad_v[index] = grad_h * one_sub_reciprocal_tau;
      sum_t_grad_h_to_rtau += grad_h * grad_h_to_rtau[mem_index];
    }
    sdata[threadIdx.x] = sum_t_grad_h_to_rtau;
  }
  else
  {
    sdata[threadIdx.x] = 0.0f;
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}

__global__ void ParametricLIF_detach_x_bptt_cuda_kernel_half(
  at::Half* __restrict__ grad_x_seq, at::Half* __restrict__ grad_v, at::Half* __restrict__ grad_rtau,
  const at::Half* __restrict__ grad_spike_seq, const at::Half* __restrict__ grad_s_to_h, const at::Half* __restrict__ grad_v_to_h, const at::Half* __restrict__ grad_h_to_rtau,
  const int neuron_num, const int size,
  const half one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ half sdata[THREADS];
  if (index < neuron_num)
  {
    half grad_h;
    half sum_t_grad_h_to_rtau = __float2half(0.0f);
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = __hfma(grad_spike_seq[mem_index], grad_s_to_h[mem_index], __hmul(grad_v[index], grad_v_to_h[mem_index]));
      grad_x_seq[mem_index] = grad_h;
      grad_v[index] = __hmul(grad_h, one_sub_reciprocal_tau);
      sum_t_grad_h_to_rtau = __hfma(grad_h, grad_h_to_rtau[mem_index], sum_t_grad_h_to_rtau);
    }
    sdata[threadIdx.x] = sum_t_grad_h_to_rtau;
  }
  else
  {
    sdata[threadIdx.x] = __float2half(0.0f);
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] = __hadd(sdata[threadIdx.x + stride], sdata[threadIdx.x]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    grad_rtau[0] = sdata[0];
  }
}
std::vector<at::Tensor> ParametricLIF_bptt(
  torch::Tensor & grad_spike_seq, torch::Tensor & grad_v_next,
  torch::Tensor & grad_s_to_h, torch::Tensor & grad_v_to_h, torch::Tensor & grad_h_to_rtau,
  const float & reciprocal_tau, const bool & detach_x)
{
  CHECK_TENSOR(grad_spike_seq);
  CHECK_TENSOR(grad_v_next);
  CHECK_TENSOR(grad_s_to_h);
  CHECK_TENSOR(grad_v_to_h);
  CHECK_TENSOR(grad_h_to_rtau);
  auto grad_x_seq = torch::zeros_like(grad_spike_seq.data());
  auto grad_v = grad_v_next.data().clone();
  auto grad_rtau = torch::zeros({1}).to(grad_x_seq);
  CHECK_TENSOR(grad_x_seq);
  CHECK_TENSOR(grad_v);
  CHECK_TENSOR(grad_rtau);

  CHECK_CUDA_OPERATION(cudaSetDevice(grad_spike_seq.get_device()));
  const int seq_len = grad_spike_seq.size(0);
  const int size = grad_spike_seq.numel();
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  if (grad_x_seq.scalar_type() == c10::ScalarType::Float)
  {
    if (detach_x)
    {
      ParametricLIF_detach_x_bptt_cuda_kernel<<<blocks, threads>>>(
        grad_x_seq.data_ptr<float>(), grad_v.data_ptr<float>(), grad_rtau.data_ptr<float>(),
        grad_spike_seq.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
        neuron_num, size,
        1.0f - reciprocal_tau);
    }
    else
    {
      ParametricLIF_bptt_cuda_kernel<<<blocks, threads>>>(
        grad_x_seq.data_ptr<float>(), grad_v.data_ptr<float>(), grad_rtau.data_ptr<float>(),
        grad_spike_seq.data_ptr<float>(), grad_s_to_h.data_ptr<float>(), grad_v_to_h.data_ptr<float>(), grad_h_to_rtau.data_ptr<float>(),
        neuron_num, size,
        reciprocal_tau, 1.0f - reciprocal_tau);
    }

  }
  else if (grad_x_seq.scalar_type() == c10::ScalarType::Half)
  {
    if (detach_x)
    {
      ParametricLIF_detach_x_bptt_cuda_kernel_half<<<blocks, threads>>>(
        grad_x_seq.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(), grad_rtau.data_ptr<at::Half>(),
        grad_spike_seq.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
        neuron_num, size,
        __float2half(1.0f - reciprocal_tau));
    }
    else
    {
      ParametricLIF_bptt_cuda_kernel_half<<<blocks, threads>>>(
        grad_x_seq.data_ptr<at::Half>(), grad_v.data_ptr<at::Half>(), grad_rtau.data_ptr<at::Half>(),
        grad_spike_seq.data_ptr<at::Half>(), grad_s_to_h.data_ptr<at::Half>(), grad_v_to_h.data_ptr<at::Half>(), grad_h_to_rtau.data_ptr<at::Half>(),
        neuron_num, size,
        __float2half(reciprocal_tau), __float2half(1.0f - reciprocal_tau));
    }

  }


  return {grad_x_seq, grad_v, grad_rtau};
}