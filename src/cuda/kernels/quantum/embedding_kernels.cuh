#ifndef SEP_CUDA_EMBEDDING_KERNELS_CUH
#define SEP_CUDA_EMBEDDING_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "common/kernel_launch.h"
#include "common/memory/device_buffer.h"
#include "common/stream/stream.h"

namespace sep {
namespace cuda {
namespace quantum {

// Embedding similarity calculator kernel
// Computes dot product similarity between embeddings
__global__ void similarity_kernel(
    float* d_similarity,
    const float* d_emb_a,
    const float* d_emb_b,
    std::uint32_t embedding_size
);

// Embedding blending kernel
// Combines multiple embeddings with weights
__global__ void blend_kernel(
    float* d_output,
    const float* d_embeddings,
    const float* d_weights,
    std::uint32_t num_contexts,
    std::uint32_t embedding_size
);

// Similarity kernel launch wrapper using Buffer abstractions
cudaError_t launchSimilarityKernel(
    DeviceBuffer<float>& similarity,
    const DeviceBuffer<float>& embedding_a,
    const DeviceBuffer<float>& embedding_b,
    const Stream& stream = Stream()
);

// Blend kernel launch wrapper using Buffer abstractions
cudaError_t launchBlendKernel(
    DeviceBuffer<float>& output,
    const DeviceBuffer<float>& embeddings,
    const DeviceBuffer<float>& weights,
    std::uint32_t num_contexts,
    const Stream& stream = Stream()
);

}}} // namespace sep::cuda::quantum

#endif // SEP_CUDA_EMBEDDING_KERNELS_CUH