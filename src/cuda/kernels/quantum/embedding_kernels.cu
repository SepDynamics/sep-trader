#include <cuda_runtime.h>

#include <cstdint>

#include "common/error/cuda_error.h"
#include "embedding_kernels.cuh"

namespace sep {
namespace cuda {
namespace quantum {

__global__ void similarity_kernel(
    float* d_similarity,
    const float* d_emb_a,
    const float* d_emb_b,
    std::uint32_t embedding_size
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= embedding_size)
        return;

    float dot_product = 0.0f;
    for (std::uint32_t i = 0; i < embedding_size; ++i) {
        dot_product += d_emb_a[i] * d_emb_b[i];
    }

    if (tid == 0) {
        *d_similarity = dot_product;
    }
}

__global__ void blend_kernel(
    float* d_output,
    const float* d_embeddings,
    const float* d_weights,
    std::uint32_t num_contexts,
    std::uint32_t embedding_size
) {
    const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= embedding_size)
        return;

    float sum = 0.0f;
    for (std::uint32_t i = 0; i < num_contexts; ++i) {
        sum += d_embeddings[i * embedding_size + tid] * d_weights[i];
    }

    d_output[tid] = sum;
}

cudaError_t launchSimilarityKernel(
    DeviceBuffer<float>& similarity,
    const DeviceBuffer<float>& embedding_a,
    const DeviceBuffer<float>& embedding_b,
    const Stream& stream
) {
    // Validate input parameters
    const std::uint32_t embedding_size = embedding_a.size();
    
    if (embedding_b.size() != embedding_size) {
        return cudaErrorInvalidValue;
    }
    
    if (similarity.size() != 1) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const std::uint32_t block_size = 256;
    const std::uint32_t grid_size = (embedding_size + block_size - 1) / block_size;
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    similarity_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        similarity.data(),
        embedding_a.data(),
        embedding_b.data(),
        embedding_size
    );
    
    return cudaGetLastError();
}

cudaError_t launchBlendKernel(
    DeviceBuffer<float>& output,
    const DeviceBuffer<float>& embeddings,
    const DeviceBuffer<float>& weights,
    std::uint32_t num_contexts,
    const Stream& stream
) {
    // Validate input parameters
    const std::uint32_t embedding_size = output.size();
    
    if (embeddings.size() != embedding_size * num_contexts) {
        return cudaErrorInvalidValue;
    }
    
    if (weights.size() != num_contexts) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const std::uint32_t block_size = 256;
    const std::uint32_t grid_size = (embedding_size + block_size - 1) / block_size;
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    blend_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        output.data(),
        embeddings.data(),
        weights.data(),
        num_contexts,
        embedding_size
    );
    
    return cudaGetLastError();
}

}}} // namespace sep::cuda::quantum