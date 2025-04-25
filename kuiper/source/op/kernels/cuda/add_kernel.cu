#include "add_kernel.cuh"

#define FLOAT4(val) (reinterpret_cast<float4*>(&(val))[0])
namespace kernel {
__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  float in_val1 = in1[tid];
  float in_val2 = in2[tid];
  out[tid] = in_val1 + in_val2;
}

__global__ void add_kernel_cu_fp32x4(int32_t size, float* in1, float* in2, float* out){
  int32_t tid = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
  if(tid >= size){
    return;
  }
  if (tid + 3 >= size) return;

  float4 in1_float4 = FLOAT4(in1[tid]);
  float4 in2_float4 = FLOAT4(in2[tid]);
  float4 out_float4;
  out_float4.x = in1_float4.x + in2_float4.x;
  out_float4.y = in1_float4.y + in2_float4.y;
  out_float4.z = in1_float4.z + in2_float4.z;
  out_float4.w = in1_float4.w + in2_float4.w;
  FLOAT4(out[tid]) = out_float4; 
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(), input2.ptr<float>(),
                                                  const_cast<float*>(output.ptr<float>()));
  }
}

void add_kernel_cu_fp32x4(const tensor::Tensor& input1, const tensor::Tensor& input2,
  const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  if (stream) {
  cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
  add_kernel_cu_fp32x4<<<block_num / 4, thread_num, 0, stream_>>>(
  size, const_cast<float*>(input1.ptr<float>()), const_cast<float*>(input2.ptr<float>()), const_cast<float*>(output.ptr<float>()));
  } else {
  add_kernel_cu_fp32x4<<<block_num / 4, thread_num>>>(size, const_cast<float*>(input1.ptr<float>()), const_cast<float*>(input2.ptr<float>()),
                                  const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel
