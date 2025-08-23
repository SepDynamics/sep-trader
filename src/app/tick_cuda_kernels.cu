#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

// C++ standard headers
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdio.h>
#include <iostream>

// CUDA printf for device code
extern "C" {
    int printf(const char*, ...);
}

// CUDA-compatible iostream replacement
#ifdef __CUDACC__
#define CUDA_COUT(...) printf(__VA_ARGS__)
#define CUDA_CERR(...) fprintf(stderr, __VA_ARGS__)
#else
#define CUDA_COUT(...) std::cout << __VA_ARGS__
#define CUDA_CERR(...) std::cerr << __VA_ARGS__
#endif

// Project headers
#include "app/tick_cuda_kernels.cuh"
#include "app/cuda_types.cuh"

// CUDA math functions
using ::sqrtf;

namespace sep::apps::cuda {

// Forward window calculation kernel
__global__ void calculateForwardWindowsKernel(
    const TickData* ticks,
    size_t tick_count,
    ForwardWindowResult* results,
    size_t result_count,
    uint64_t window_size_ns) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result_count) return;

    ForwardWindowResult& result = results[idx];
    size_t window_start = idx * (tick_count / result_count);
    size_t window_end = (idx + 1) * (tick_count / result_count);
    
    // Initialize result
    result.coherence = 0.0f;
    result.stability = 0.0f;
    result.entropy = 0.0f;
    result.rupture_count = 0;
    result.flip_count = 0;
    result.confidence = 0.0f;
    result.converged = false;
    result.iterations = 0;
    
    // Calculate window statistics
    if (window_end > window_start + 2) {
        double mean_price = 0.0;
        size_t count = 0;
        
        // First pass - calculate mean
        for (size_t j = window_start; j < window_end; ++j) {
            mean_price += ticks[j].price;
            count++;
        }
        mean_price /= count;
        
        // Second pass - calculate volatility and detect patterns
        double volatility = 0.0;
        int direction_changes = 0;
        double prev_change = 0.0;
        
        for (size_t j = window_start + 1; j < window_end; ++j) {
            // Volatility calculation
            double diff = ticks[j].price - mean_price;
            volatility += diff * diff;
            
            // Direction change detection
            double change = ticks[j].price - ticks[j-1].price;
            if (j > window_start + 1) {
                if ((change > 0 && prev_change < 0) || (change < 0 && prev_change > 0)) {
                    direction_changes++;
                }
            }
            prev_change = change;
        }
        
        volatility = sqrt(volatility / count);
        
        // Calculate metrics
        result.coherence = static_cast<float>(1.0 - (direction_changes / static_cast<double>(count)));
        result.stability = static_cast<float>(1.0 / (1.0 + volatility));
        result.flip_count = direction_changes;
        result.confidence = result.coherence * result.stability;
        result.iterations = static_cast<int>(count);
        result.converged = (result.confidence > 0.5f);
    }
}

namespace {
// Helper functions for device memory management
template<typename T>
cudaError_t allocateDeviceMemory(T** dev_ptr, size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(dev_ptr), count * sizeof(T));
}

template<typename T>
cudaError_t copyToDevice(T* dev_ptr, const T* host_ptr, size_t count) {
    return cudaMemcpy(dev_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t copyToHost(T* host_ptr, const T* dev_ptr, size_t count) {
    return cudaMemcpy(host_ptr, dev_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
}
} // anonymous namespace

cudaError_t calculateWindowsCuda(
    CudaContext& context,
    const TickData* host_ticks, size_t tick_count,
    WindowResult* hourly_results, size_t hourly_count,
    WindowResult* daily_results, size_t daily_count,
    uint64_t current_time,
    uint64_t hourly_window_ns,
    uint64_t daily_window_ns) {
    
    if (!context.initialized) {
        return cudaErrorNotReady;
    }
    
    cudaError_t error = cudaSuccess;
    
    // Limit tick count to maximum history
    if (tick_count > MAX_TICK_HISTORY) {
        tick_count = MAX_TICK_HISTORY;
    }
    
    // Copy tick data to device
    error = cudaMemcpyAsync(context.d_ticks, host_ticks,
                           tick_count * sizeof(TickData),
                           cudaMemcpyHostToDevice, context.stream);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Error copying ticks to device: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    // Calculate grid and block dimensions
    size_t result_count = std::min(hourly_count, daily_count);
    int block_size = CUDA_BLOCK_SIZE;
    int grid_size = (result_count + block_size - 1) / block_size;
    
    // Launch multi-timeframe kernel
    calculateMultiTimeframeWindows<<<grid_size, block_size, 0, context.stream>>>(
        context.d_ticks,
        tick_count,
        context.d_hourly_results,
        context.d_daily_results,
        result_count,
        current_time,
        HOURLY_STEP_NS,
        DAILY_STEP_NS,
        hourly_window_ns,
        daily_window_ns
    );
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Kernel launch error: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    // Copy results back to host
    error = cudaMemcpyAsync(hourly_results, context.d_hourly_results,
                           result_count * sizeof(WindowResult),
                           cudaMemcpyDeviceToHost, context.stream);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Error copying hourly results: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    error = cudaMemcpyAsync(daily_results, context.d_daily_results,
                           result_count * sizeof(WindowResult),
                           cudaMemcpyDeviceToHost, context.stream);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Error copying daily results: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    // Synchronize stream
    error = cudaStreamSynchronize(context.stream);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Stream sync error: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    return cudaSuccess;
}

cudaError_t calculateForwardWindowsCuda(
    CudaContext& context,
    const TickData* ticks, size_t tick_count,
    ForwardWindowResult* results, size_t result_count,
    uint64_t window_size_ns) {
    
    if (!context.initialized) {
        return cudaErrorNotReady;
    }
    
    if (tick_count == 0 || result_count == 0) {
        return cudaSuccess;
    }
    
    // Copy input data to device
    cudaError_t error = cudaMemcpyAsync(context.d_ticks, ticks,
                                       tick_count * sizeof(TickData),
                                       cudaMemcpyHostToDevice, context.stream);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Error copying ticks to device: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    // Calculate grid dimensions
    int block_size = CUDA_BLOCK_SIZE;
    int grid_size = (result_count + block_size - 1) / block_size;
    
    // Launch kernel
    calculateForwardWindowsKernel<<<grid_size, block_size, 0, context.stream>>>(
        context.d_ticks,
        tick_count,
        results,
        result_count,
        window_size_ns
    );
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Kernel launch error: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    // Synchronize stream
    error = cudaStreamSynchronize(context.stream);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Stream sync error: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    return cudaSuccess;
}

// CUDA kernel for calculating rolling window statistics
__global__ void calculateRollingWindowsKernel(
    const TickData* ticks,
    size_t tick_count,
    const WindowSpec* window_specs,
    size_t window_count,
    WindowResult* results,
    uint64_t* window_timestamps) {
    
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (window_idx >= window_count) return;
    
    const WindowSpec& spec = window_specs[window_idx];
    WindowResult& result = results[window_idx];
    
    // Initialize result
    result.mean_price = 0.0;
    result.volatility = 0.0;
    result.price_change = 0.0;
    result.pip_change = 0.0;
    result.tick_count = 0;
    result.window_start = spec.start_time;
    result.window_end = spec.end_time;
    
    // Find ticks within this window
    double price_sum = 0.0;
    double first_price = 0.0;
    double last_price = 0.0;
    bool first_set = false;
    size_t valid_ticks = 0;
    
    for (size_t i = 0; i < tick_count; ++i) {
        const TickData& tick = ticks[i];
        
        if (tick.timestamp >= spec.start_time && tick.timestamp <= spec.end_time) {
            price_sum += tick.price;
            valid_ticks++;
            
            if (!first_set) {
                first_price = tick.price;
                first_set = true;
            }
            last_price = tick.price;
        }
    }
    
    result.tick_count = valid_ticks;
    
    if (valid_ticks > 0) {
        result.mean_price = price_sum / valid_ticks;
        
        // Calculate volatility (standard deviation)
        double variance_sum = 0.0;
        for (size_t i = 0; i < tick_count; ++i) {
            const TickData& tick = ticks[i];
            
            if (tick.timestamp >= spec.start_time && tick.timestamp <= spec.end_time) {
                double diff = tick.price - result.mean_price;
                variance_sum += diff * diff;
            }
        }

        result.volatility = sqrt(variance_sum / valid_ticks);
        result.price_change = last_price - first_price;
        result.pip_change = result.price_change * 10000.0; // Convert to pips
    }
}

// Optimized kernel using shared memory for better performance
__global__ void calculateRollingWindowsOptimized(
    const TickData* ticks,
    size_t tick_count,
    const WindowSpec* window_specs,
    size_t window_count,
    WindowResult* results,
    uint64_t current_time,
    uint64_t hourly_window_ns,
    uint64_t daily_window_ns) {
    
    __shared__ double shared_prices[CUDA_BLOCK_SIZE];
    __shared__ uint64_t shared_timestamps[CUDA_BLOCK_SIZE];
    
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;
    
    if (window_idx >= window_count) return;
    
    // Determine window type and calculate bounds
    bool is_hourly = (window_idx % 2 == 0);
    uint64_t window_start = current_time - (is_hourly ? hourly_window_ns : daily_window_ns);
    uint64_t window_end = current_time;
    
    WindowResult& result = results[window_idx];
    result.window_start = window_start;
    result.window_end = window_end;
    result.tick_count = 0;
    result.mean_price = 0.0;
    result.volatility = 0.0;
    
    // Process ticks in chunks using shared memory
    double price_sum = 0.0;
    double first_price = 0.0;
    double last_price = 0.0;
    bool first_set = false;
    size_t valid_ticks = 0;
    
    for (size_t chunk_start = 0; chunk_start < tick_count; chunk_start += CUDA_BLOCK_SIZE) {
        // Load chunk into shared memory
        size_t load_idx = chunk_start + thread_id;
        if (load_idx < tick_count) {
            shared_prices[thread_id] = ticks[load_idx].price;
            shared_timestamps[thread_id] = ticks[load_idx].timestamp;
        } else {
            shared_timestamps[thread_id] = 0; // Invalid timestamp
        }
        
        __syncthreads();
        
        // Process shared memory data (only thread 0 for each window)
        if (thread_id == 0) {
            for (int i = 0; i < CUDA_BLOCK_SIZE && (chunk_start + i) < tick_count; ++i) {
                if (shared_timestamps[i] >= window_start && shared_timestamps[i] <= window_end) {
                    price_sum += shared_prices[i];
                    valid_ticks++;
                    
                    if (!first_set) {
                        first_price = shared_prices[i];
                        first_set = true;
                    }
                    last_price = shared_prices[i];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Finalize results (only thread 0)
    if (thread_id == 0) {
        result.tick_count = valid_ticks;
        
        if (valid_ticks > 0) {
            result.mean_price = price_sum / valid_ticks;
            result.price_change = last_price - first_price;
            result.pip_change = result.price_change * 10000.0;
            
            // Calculate volatility in second pass
            double variance_sum = 0.0;
            for (size_t chunk_start = 0; chunk_start < tick_count; chunk_start += CUDA_BLOCK_SIZE) {
                for (size_t i = chunk_start; i < (chunk_start + CUDA_BLOCK_SIZE < tick_count ? chunk_start + CUDA_BLOCK_SIZE : tick_count); ++i) {
                    if (ticks[i].timestamp >= window_start && ticks[i].timestamp <= window_end) {
                        double diff = ticks[i].price - result.mean_price;
                        variance_sum += diff * diff;
                    }
                }
            }
            result.volatility = sqrtf(static_cast<float>(variance_sum / valid_ticks));
        }
    }
}

// Multi-timeframe parallel processing kernel
__global__ void calculateMultiTimeframeWindows(
    const TickData* ticks,
    size_t tick_count,
    WindowResult* hourly_results,
    WindowResult* daily_results,
    size_t result_count,
    uint64_t base_time,
    uint64_t hourly_step_ns,
    uint64_t daily_step_ns,
    uint64_t hourly_window_ns,
    uint64_t daily_window_ns) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= result_count) return;
    
    // Calculate hourly window
    uint64_t hourly_end = base_time - (idx * hourly_step_ns);
    uint64_t hourly_start = hourly_end - hourly_window_ns;
    
    WindowResult& hourly_result = hourly_results[idx];
    calculateWindowStats(ticks, tick_count, hourly_start, hourly_end, hourly_result);
    
    // Calculate daily window
    uint64_t daily_end = base_time - (idx * daily_step_ns);
    uint64_t daily_start = daily_end - daily_window_ns;
    
    WindowResult& daily_result = daily_results[idx];
    calculateWindowStats(ticks, tick_count, daily_start, daily_end, daily_result);
}

__device__ void calculateWindowStats(
    const TickData* ticks,
    size_t tick_count,
    uint64_t window_start,
    uint64_t window_end,
    WindowResult& result) {
    
    result.window_start = window_start;
    result.window_end = window_end;
    result.tick_count = 0;
    
    double price_sum = 0.0;
    double first_price = 0.0;
    double last_price = 0.0;
    bool first_set = false;
    size_t valid_ticks = 0;
    
    // First pass: count, sum, find first/last
    for (size_t i = 0; i < tick_count; ++i) {
        const TickData& tick = ticks[i];
        
        if (tick.timestamp >= window_start && tick.timestamp <= window_end) {
            price_sum += tick.price;
            valid_ticks++;
            
            if (!first_set) {
                first_price = tick.price;
                first_set = true;
            }
            last_price = tick.price;
        }
    }
    
    result.tick_count = valid_ticks;
    
    if (valid_ticks > 0) {
        result.mean_price = price_sum / valid_ticks;
        result.price_change = last_price - first_price;
        result.pip_change = result.price_change * 10000.0;
        
        // Second pass: calculate volatility
        double variance_sum = 0.0;
        for (size_t i = 0; i < tick_count; ++i) {
            const TickData& tick = ticks[i];
            
            if (tick.timestamp >= window_start && tick.timestamp <= window_end) {
                double diff = tick.price - result.mean_price;
                variance_sum += diff * diff;
            }
        }
        
        result.volatility = sqrt(variance_sum / valid_ticks);
    } else {
        result.mean_price = 0.0;
        result.price_change = 0.0;
        result.pip_change = 0.0;
        result.volatility = 0.0;
    }
}

// Device management function implementations
cudaError_t initializeCudaDevice(CudaContext& context, int device_id) {
    cudaError_t error = cudaSuccess;
    
    // Set device
    error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Failed to set device %d: %s\n", device_id, cudaGetErrorString(error));
        return error;
    }
    
    // Create CUDA stream
    error = cudaStreamCreate(&context.stream);
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Failed to create stream: %s\n", cudaGetErrorString(error));
        return error;
    }
    
    // Allocate device memory for tick data
    error = cudaMalloc(&context.d_ticks, MAX_TICK_HISTORY * sizeof(TickData));
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Failed to allocate device memory for ticks: %s\n", cudaGetErrorString(error));
        cudaStreamDestroy(context.stream);
        return error;
    }
    
    // Allocate device memory for hourly results
    error = cudaMalloc(&context.d_hourly_results, MAX_CALCULATION_WINDOWS * sizeof(WindowResult));
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Failed to allocate device memory for hourly results: %s\n", cudaGetErrorString(error));
        cudaFree(context.d_ticks);
        cudaStreamDestroy(context.stream);
        return error;
    }
    
    // Allocate device memory for daily results
    error = cudaMalloc(&context.d_daily_results, MAX_CALCULATION_WINDOWS * sizeof(WindowResult));
    if (error != cudaSuccess) {
        CUDA_CERR("[CUDA] Failed to allocate device memory for daily results: %s\n", cudaGetErrorString(error));
        cudaFree(context.d_ticks);
        cudaFree(context.d_hourly_results);
        cudaStreamDestroy(context.stream);
        return error;
    }
    
    // Mark context as initialized
    context.initialized = true;
    
    printf("[CUDA] Device %d initialized successfully\n", device_id);
    return cudaSuccess;
}

cudaError_t cleanupCudaDevice(CudaContext& context) {
    if (!context.initialized) {
        return cudaSuccess;
    }
    
    cudaError_t error = cudaSuccess;
    
    // Free device memory
    if (context.d_ticks) {
        cudaFree(context.d_ticks);
        context.d_ticks = nullptr;
    }
    
    if (context.d_hourly_results) {
        cudaFree(context.d_hourly_results);
        context.d_hourly_results = nullptr;
    }
    
    if (context.d_daily_results) {
        cudaFree(context.d_daily_results);
        context.d_daily_results = nullptr;
    }
    
    // Destroy CUDA stream
    if (context.stream) {
        error = cudaStreamDestroy(context.stream);
        context.stream = nullptr;
    }
    
    // Mark context as uninitialized
    context.initialized = false;
    
    printf("[CUDA] Device cleaned up successfully\n");
    return error;
}

} // namespace sep::apps::cuda