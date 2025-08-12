#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>
#include <complex>

#include "../../common/kernel_launch.h"
#include "../../common/stream/stream.h"
#include "../../common/memory/device_buffer.h"
#include "../../common/error/cuda_error.h"

#include "quantum_types.cuh"

namespace sep {
namespace cuda {
namespace quantum {

// Define complex type for clarity
using ComplexType = cufftComplex;

// Kernel for normalizing FFT results
__global__ void normalize_fft_kernel(
    ComplexType* d_fft_data,
    uint32_t fft_size,
    uint32_t num_patterns
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pattern_idx = tid / fft_size;
    const uint32_t fft_idx = tid % fft_size;
    
    if (pattern_idx >= num_patterns || fft_idx >= fft_size)
        return;
    
    // Normalize by FFT size
    const float scale = 1.0f / sqrtf(static_cast<float>(fft_size));
    const uint32_t idx = pattern_idx * fft_size + fft_idx;
    
    d_fft_data[idx].x *= scale;
    d_fft_data[idx].y *= scale;
}

// Kernel for computing cross-spectrum between FFT results
__global__ void cross_spectrum_kernel(
    const ComplexType* d_fft_data1,
    const ComplexType* d_fft_data2,
    ComplexType* d_cross_spectrum,
    uint32_t fft_size,
    uint32_t num_patterns
) {
    const uint32_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t fft_idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (tid_x >= num_patterns || tid_y >= num_patterns || fft_idx >= fft_size)
        return;
    
    // Only compute upper triangle (symmetry)
    if (tid_y > tid_x)
        return;
    
    // Calculate indices
    const uint32_t idx1 = tid_x * fft_size + fft_idx;
    const uint32_t idx2 = tid_y * fft_size + fft_idx;
    const uint32_t out_idx = (tid_x * num_patterns + tid_y) * fft_size + fft_idx;
    
    // Compute cross spectrum: F1 * conj(F2)
    const float real1 = d_fft_data1[idx1].x;
    const float imag1 = d_fft_data1[idx1].y;
    const float real2 = d_fft_data2[idx2].x;
    const float imag2 = d_fft_data2[idx2].y;
    
    // Complex multiplication with conjugate of second term
    d_cross_spectrum[out_idx].x = real1 * real2 + imag1 * imag2;
    d_cross_spectrum[out_idx].y = imag1 * real2 - real1 * imag2;
    
    // Store in symmetric position if different patterns
    if (tid_x != tid_y) {
        const uint32_t sym_idx = (tid_y * num_patterns + tid_x) * fft_size + fft_idx;
        // Conjugate for the symmetric position
        d_cross_spectrum[sym_idx].x = d_cross_spectrum[out_idx].x;
        d_cross_spectrum[sym_idx].y = -d_cross_spectrum[out_idx].y;
    }
}

// Kernel for calculating QFH coherence from cross-spectrum
__global__ void qfh_coherence_kernel(
    const ComplexType* d_cross_spectrum,
    float* d_coherence_matrix,
    uint32_t fft_size,
    uint32_t num_patterns
) {
    const uint32_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tid_x >= num_patterns || tid_y >= num_patterns)
        return;
    
    // Only compute upper triangle (symmetry)
    if (tid_y > tid_x)
        return;
    
    float coherence_sum = 0.0f;
    
    // Sum magnitudes of cross-spectrum components
    for (uint32_t i = 0; i < fft_size; ++i) {
        const uint32_t idx = (tid_x * num_patterns + tid_y) * fft_size + i;
        const float real = d_cross_spectrum[idx].x;
        const float imag = d_cross_spectrum[idx].y;
        
        // Add magnitude squared
        coherence_sum += real * real + imag * imag;
    }
    
    // Normalize by FFT size
    coherence_sum /= fft_size;
    
    // Store coherence in matrix (both positions for symmetry)
    d_coherence_matrix[tid_y * num_patterns + tid_x] = coherence_sum;
    if (tid_x != tid_y) {
        d_coherence_matrix[tid_x * num_patterns + tid_y] = coherence_sum;
    }
}

// Launch wrapper for normalizing FFT results
cudaError_t launchNormalizeFftKernel(
    DeviceBuffer<ComplexType>& fft_data,
    uint32_t fft_size,
    uint32_t num_patterns,
    const Stream& stream
) {
    // Validate input parameters
    if (fft_data.size() != fft_size * num_patterns) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const uint32_t total_elements = fft_size * num_patterns;
    const uint32_t block_size = 256;
    const uint32_t grid_size = (total_elements + block_size - 1) / block_size;
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    normalize_fft_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        fft_data.data(),
        fft_size,
        num_patterns
    );
    
    return cudaGetLastError();
}

// Launch wrapper for computing cross-spectrum
cudaError_t launchCrossSpectrumKernel(
    const DeviceBuffer<ComplexType>& fft_data1,
    const DeviceBuffer<ComplexType>& fft_data2,
    DeviceBuffer<ComplexType>& cross_spectrum,
    uint32_t fft_size,
    uint32_t num_patterns,
    const Stream& stream
) {
    // Validate input parameters
    if (fft_data1.size() != fft_size * num_patterns ||
        fft_data2.size() != fft_size * num_patterns ||
        cross_spectrum.size() != fft_size * num_patterns * num_patterns) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const dim3 block_size(8, 8, 8);
    const dim3 grid_size(
        (num_patterns + block_size.x - 1) / block_size.x,
        (num_patterns + block_size.y - 1) / block_size.y,
        (fft_size + block_size.z - 1) / block_size.z
    );
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    cross_spectrum_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        fft_data1.data(),
        fft_data2.data(),
        cross_spectrum.data(),
        fft_size,
        num_patterns
    );
    
    return cudaGetLastError();
}

// Launch wrapper for QFH coherence calculation
cudaError_t launchQfhCoherenceKernel(
    const DeviceBuffer<ComplexType>& cross_spectrum,
    DeviceBuffer<float>& coherence_matrix,
    uint32_t fft_size,
    uint32_t num_patterns,
    const Stream& stream
) {
    // Validate input parameters
    if (cross_spectrum.size() != fft_size * num_patterns * num_patterns ||
        coherence_matrix.size() != num_patterns * num_patterns) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate launch configuration
    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (num_patterns + block_size.x - 1) / block_size.x,
        (num_patterns + block_size.y - 1) / block_size.y
    );
    
    LaunchConfig config(grid_size, block_size, 0, stream.get());
    
    // Launch the kernel
    qfh_coherence_kernel<<<config.grid, config.block, config.shared_memory_bytes, config.stream>>>(
        cross_spectrum.data(),
        coherence_matrix.data(),
        fft_size,
        num_patterns
    );
    
    return cudaGetLastError();
}

// Helper function to perform FFT using cuFFT
cudaError_t performFFT(
    const DeviceBuffer<float>& input_data,
    DeviceBuffer<ComplexType>& output_fft,
    uint32_t signal_size,
    uint32_t num_signals,
    const Stream& stream
) {
    // Validate input parameters
    if (input_data.size() != signal_size * num_signals ||
        output_fft.size() != signal_size * num_signals) {
        return cudaErrorInvalidValue;
    }
    
    cufftHandle plan;
    cufftResult cufft_result;
    
    // Create FFT plan
    cufft_result = cufftPlan1d(&plan, signal_size, CUFFT_R2C, num_signals);
    if (cufft_result != CUFFT_SUCCESS) {
        return cudaErrorUnknown;
    }
    
    // Set stream
    cufft_result = cufftSetStream(plan, stream.get());
    if (cufft_result != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        return cudaErrorUnknown;
    }
    
    // Execute FFT
    cufft_result = cufftExecR2C(plan, 
                               const_cast<float*>(input_data.data()), 
                               reinterpret_cast<cufftComplex*>(output_fft.data()));
    
    // Clean up
    cufftDestroy(plan);
    
    return (cufft_result == CUFFT_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

// Main function to perform Quantum Fourier Hierarchy analysis
cudaError_t performQFHAnalysis(
    const DeviceBuffer<float>& pattern_data,
    DeviceBuffer<ComplexType>& fft_results,
    DeviceBuffer<ComplexType>& cross_spectrum,
    DeviceBuffer<float>& coherence_matrix,
    uint32_t pattern_size,
    uint32_t num_patterns,
    const Stream& stream
) {
    cudaError_t cuda_error;
    
    // Step 1: Perform FFT on pattern data
    cuda_error = performFFT(pattern_data, fft_results, pattern_size, num_patterns, stream);
    if (cuda_error != cudaSuccess) {
        return cuda_error;
    }
    
    // Step 2: Normalize FFT results
    cuda_error = launchNormalizeFftKernel(fft_results, pattern_size, num_patterns, stream);
    if (cuda_error != cudaSuccess) {
        return cuda_error;
    }
    
    // Step 3: Compute cross-spectrum
    cuda_error = launchCrossSpectrumKernel(fft_results, fft_results, cross_spectrum, pattern_size, num_patterns, stream);
    if (cuda_error != cudaSuccess) {
        return cuda_error;
    }
    
    // Step 4: Calculate QFH coherence
    cuda_error = launchQfhCoherenceKernel(cross_spectrum, coherence_matrix, pattern_size, num_patterns, stream);
    
    return cuda_error;
}

} // namespace quantum
} // namespace cuda
} // namespace sep