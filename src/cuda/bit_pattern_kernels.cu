#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm> // For std::min, std::max
#include <cmath>     // For log2, fabs

#include "bit_pattern_types.cuh" // Include the new device types
#include "util/result.h"
#include "error_handler.h"

// Helper device functions (will be ported from forward_window_kernels.cpp)
__device__ bool detectTrendAcceleration(const uint8_t* window, size_t window_size);
__device__ bool detectMeanReversion(const uint8_t* window, size_t window_size);
__device__ bool detectVolatilityBreakout(const uint8_t* window, size_t window_size);

// CUDA kernel to analyze bit patterns
__global__ void analyzeBitPatternsKernel(const uint8_t* d_bits,
                                         size_t total_bits_size,
                                         size_t index_start,
                                         size_t window_size,
                                         sep::apps::cuda::ForwardWindowResultDevice* d_results) {
    // Each thread processes one window, but for simplicity, we'll assume one window for now
    // This kernel needs to be adapted for batch processing if multiple windows are to be processed in parallel.
    // For now, we'll assume a single window is passed and processed by thread 0.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sep::apps::cuda::ForwardWindowResultDevice result;
        result.flip_count = 0;
        result.rupture_count = 0;
        result.entropy = 0.0f;
        result.coherence = 0.0f;
        result.stability = 0.0f;
        result.confidence = 0.0f;

        if (total_bits_size <= index_start + 1) {
            d_results[0] = result;
            return;
        }

        // Create a local window for processing
        // This is a simplification; for larger windows, this would need to be optimized
        // (e.g., shared memory, or processing directly from global memory)
        uint8_t local_window[10]; // Assuming max window_size is 10 for now
        for (size_t i = 0; i < window_size; ++i) {
            local_window[i] = d_bits[index_start + i];
        }

        // Calculate flip and rupture counts
        for (size_t i = 1; i < window_size; ++i) {
            if (local_window[i-1] != local_window[i]) {
                result.flip_count++;
            } else if (local_window[i-1] == 1 && local_window[i] == 1) {
                result.rupture_count++;
            }
        }

        // Calculate entropy (Shannon entropy)
        size_t ones = 0;
        for (size_t i = 0; i < window_size; ++i) {
            if (local_window[i] == 1) ones++;
        }
        size_t zeros = window_size - ones;

        if (ones > 0 && zeros > 0) {
            double p1 = static_cast<double>(ones) / window_size;
            double p0 = static_cast<double>(zeros) / window_size;
            result.entropy = -(p1 * log2(p1) + p0 * log2(p0));
        } else {
            result.entropy = 0.0f;
        }

        // Coherence and Stability logic (simplified for now, will port full logic later)
        // This part will be replaced by the actual pattern detection logic
        result.coherence = 0.5f; // Placeholder
        result.stability = 0.5f; // Placeholder

        // Set confidence based on window size and pattern consistency
        result.confidence = fminf(1.0f, static_cast<float>(window_size) / 10.0f) * result.coherence;

        d_results[0] = result;
    }
}

// Host-side launcher function
extern "C" sep::SEPResult launchAnalyzeBitPatternsKernel(const uint8_t* h_bits,
                                                      size_t total_bits_size,
                                                      size_t index_start,
                                                      size_t window_size,
                                                      sep::apps::cuda::ForwardWindowResultDevice* h_results,
                                                      cudaStream_t stream) {
    // Allocate device memory for input bits
    uint8_t* d_bits;
    cudaError_t err = cudaMallocAsync(&d_bits, total_bits_size * sizeof(uint8_t), stream);
    if (err != cudaSuccess) {
        sep::core::ErrorHandler::instance().reportError(sep::Error(sep::SEPResult::CUDA_ERROR, "Failed to allocate device memory for bits: " + std::string(cudaGetErrorString(err))));
        return sep::SEPResult::CUDA_ERROR;
    }
    err = cudaMemcpyAsync(d_bits, h_bits, total_bits_size * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_bits, stream);
        sep::core::ErrorHandler::instance().reportError(sep::Error(sep::SEPResult::CUDA_ERROR, "Failed to copy bits to device: " + std::string(cudaGetErrorString(err))));
        return sep::SEPResult::CUDA_ERROR;
    }

    // Allocate device memory for results
    sep::apps::cuda::ForwardWindowResultDevice* d_results;
    err = cudaMallocAsync(&d_results, sizeof(sep::apps::cuda::ForwardWindowResultDevice), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_bits, stream);
        cudaFreeAsync(d_bits, stream);
        sep::core::ErrorHandler::instance().reportError(sep::Error(sep::SEPResult::CUDA_ERROR, "Failed to allocate device memory for results: " + std::string(cudaGetErrorString(err))));
        return sep::SEPResult::CUDA_ERROR;
    }

    // Launch the kernel
    // Assuming a single block and single thread for now, as the kernel is designed for a single window.
    // This will need to be adjusted for batch processing.
    analyzeBitPatternsKernel<<<1, 1, 0, stream>>>(d_bits, total_bits_size, index_start, window_size, d_results);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeAsync(d_bits, stream);
        cudaFreeAsync(d_results, stream);
        cudaFreeAsync(d_bits, stream);
        cudaFreeAsync(d_results, stream);
        sep::core::ErrorHandler::instance().reportError(sep::Error(sep::SEPResult::CUDA_ERROR, "Kernel launch failed: " + std::string(cudaGetErrorString(err))));
        return sep::SEPResult::CUDA_ERROR;
    }

    // Copy results back to host
    err = cudaMemcpyAsync(h_results, d_results, sizeof(sep::apps::cuda::ForwardWindowResultDevice), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_bits, stream);
        cudaFreeAsync(d_results, stream);
        cudaFreeAsync(d_bits, stream);
        cudaFreeAsync(d_results, stream);
        sep::core::ErrorHandler::instance().reportError(sep::Error(sep::SEPResult::CUDA_ERROR, "Failed to copy results to host: " + std::string(cudaGetErrorString(err))));
        return sep::SEPResult::CUDA_ERROR;
    }

    // Synchronize and free device memory
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_bits, stream);
        cudaFreeAsync(d_results, stream);
        cudaFreeAsync(d_bits, stream);
        cudaFreeAsync(d_results, stream);
        sep::core::ErrorHandler::instance().reportError(sep::Error(sep::SEPResult::CUDA_ERROR, "Stream synchronization failed: " + std::string(cudaGetErrorString(err))));
        return sep::SEPResult::CUDA_ERROR;
    }
    cudaFreeAsync(d_bits, stream);
    cudaFreeAsync(d_results, stream);

    return sep::SEPResult::SUCCESS;
}
