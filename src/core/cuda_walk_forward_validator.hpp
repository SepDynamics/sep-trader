#pragma once

#include "core/sep_precompiled.h"
#include "core/temporal_data_validator.hpp"
#include "io/oanda_connector.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

namespace sep::validation::cuda
{
    /**
     * CUDA Walk-Forward Validation Configuration
     * Advanced GPU-accelerated validation parameters
     */
    struct CudaWalkForwardConfig
    {
        // CUDA Device Configuration
        int cuda_device_id = 0;
        size_t cuda_threads_per_block = 256;
        size_t cuda_max_blocks = 65535;
        size_t cuda_shared_memory_per_block = 49152; // 48KB
        
        // Walk-Forward GPU Optimization
        size_t gpu_batch_size = 1024;               // Parallel validation windows
        size_t gpu_memory_pool_mb = 512;            // GPU memory pool size
        bool use_tensor_cores = true;               // Tensor core acceleration
        bool use_cuda_streams = true;               // Multi-stream processing
        size_t num_cuda_streams = 8;                // Concurrent CUDA streams
        
        // Advanced Validation Parameters
        size_t forward_window_depth = 24;           // Hours in forward validation
        size_t training_window_multiplier = 7;      // 7x forward window for training
        double gpu_convergence_threshold = 1e-8;    // GPU numerical precision
        size_t max_gpu_iterations = 10000;          // Maximum GPU kernel iterations
        
        // Volumetric Sampling Configuration
        size_t volume_sample_points = 512;          // 3D volume sampling points
        size_t volume_interpolation_order = 3;      // Cubic interpolation
        bool enable_volume_caching = true;          // Cache volume computations
        size_t volume_cache_size_mb = 128;          // Volume cache size
        
        // Performance Optimization
        bool enable_gpu_profiling = true;           // CUDA profiling
        bool use_half_precision = false;            // FP16 for speed
        bool enable_async_memcpy = true;            // Async GPU-CPU transfers
        double gpu_utilization_target = 0.95;       // Target GPU utilization
    };

    /**
     * CUDA Walk-Forward Validation Results
     * GPU-accelerated comprehensive results
     */
    struct CudaWalkForwardResult
    {
        // GPU Acceleration Metrics
        std::vector<double> gpu_accuracies;
        std::vector<double> gpu_processing_times_ms;
        double total_gpu_time_ms = 0.0;
        double average_gpu_utilization = 0.0;
        double gpu_memory_efficiency = 0.0;
        
        // Advanced Walk-Forward Results
        std::vector<std::vector<double>> batch_accuracies; // Per-batch results
        std::map<size_t, double> stream_performances;      // Per-stream results
        std::vector<double> convergence_metrics;           // Convergence tracking
        std::vector<double> numerical_stability_scores;    // Numerical stability
        
        // 3D Volumetric Sampling Results
        std::vector<std::vector<std::vector<double>>> volume_samples; // 3D volume data
        std::map<size_t, double> volume_coherence_scores;             // Volume coherence
        std::vector<double> volume_interpolation_errors;              // Interpolation accuracy
        double overall_volumetric_fidelity = 0.0;
        
        // GPU Resource Utilization
        size_t peak_gpu_memory_used_mb = 0;
        double average_gpu_temperature_celsius = 0.0;
        double gpu_power_consumption_watts = 0.0;
        std::map<std::string, double> kernel_execution_times_ms;
        
        // Performance Comparison
        double cpu_baseline_time_ms = 0.0;
        double gpu_speedup_factor = 0.0;
        double energy_efficiency_ratio = 0.0;
        
        // Error Analysis
        std::vector<std::string> cuda_warnings;
        std::vector<std::string> cuda_errors;
        bool gpu_validation_successful = false;
        std::string cuda_device_info;
    };

    /**
     * CUDA-Accelerated Walk-Forward Validator
     * High-performance GPU validation system
     */
    class CudaWalkForwardValidator
    {
    public:
        explicit CudaWalkForwardValidator(const CudaWalkForwardConfig& config = {});
        ~CudaWalkForwardValidator();

        // Primary GPU validation interface
        CudaWalkForwardResult validateWithCuda(
            const std::string& pair_symbol,
            const std::vector<sep::connectors::MarketData>& historical_data);
        
        // Async GPU validation
        std::future<CudaWalkForwardResult> validateWithCudaAsync(
            const std::string& pair_symbol,
            const std::vector<sep::connectors::MarketData>& historical_data);
        
        // Batch GPU validation (multiple pairs simultaneously)
        std::map<std::string, CudaWalkForwardResult> batchValidateWithCuda(
            const std::map<std::string, std::vector<sep::connectors::MarketData>>& pair_data);
        
        // Advanced GPU walk-forward with volumetric sampling
        CudaWalkForwardResult advancedVolumetricValidation(
            const std::vector<sep::connectors::MarketData>& market_data);
        
        // 3D Volume sampling with CUDA acceleration
        std::vector<std::vector<std::vector<double>>> perform3DVolumeGpuSampling(
            const std::vector<sep::connectors::MarketData>& market_data,
            size_t depth_levels, size_t time_buckets, size_t pattern_buckets);
        
        // GPU-accelerated coherence calculation
        double calculateGpuCoherence(
            const thrust::device_vector<float>& data1,
            const thrust::device_vector<float>& data2);
        
        // Multi-stream parallel validation
        std::vector<double> multiStreamValidation(
            const std::vector<std::vector<sep::connectors::MarketData>>& validation_batches);
        
        // Configuration and device management
        void updateConfig(const CudaWalkForwardConfig& config);
        CudaWalkForwardConfig getCurrentConfig() const;
        std::vector<std::string> getAvailableCudaDevices() const;
        bool initializeCudaDevice(int device_id);
        
        // Performance monitoring
        void enableGpuProfiling(bool enable);
        std::map<std::string, double> getGpuProfilingResults() const;
        void resetGpuProfiler();
        
        // Memory management
        void optimizeGpuMemoryUsage();
        size_t getAvailableGpuMemory() const;
        void clearGpuCache();
        
    private:
        // CUDA initialization and cleanup
        bool initializeCudaContext();
        void cleanupCudaContext();
        void initializeCudaStreams();
        void cleanupCudaStreams();
        void initializeGpuMemoryPool();
        void cleanupGpuMemoryPool();
        
        // Core CUDA kernel implementations
        void launchWalkForwardKernel(
            const float* input_data, float* output_data,
            size_t data_size, size_t window_size, size_t num_windows,
            cudaStream_t stream);
        
        void launchVolumetricSamplingKernel(
            const float* market_data, float* volume_output,
            size_t data_points, size_t depth_levels, size_t time_buckets,
            cudaStream_t stream);
        
        void launchCoherenceCalculationKernel(
            const float* data1, const float* data2, float* coherence_output,
            size_t data_size, cudaStream_t stream);
        
        // Advanced GPU algorithms
        void performGpuTensorOperations(
            const thrust::device_vector<float>& input,
            thrust::device_vector<float>& output);
        
        std::vector<double> executeMultiStreamProcessing(
            const std::vector<thrust::device_vector<float>>& stream_data);
        
        // CUDA utility functions
        void checkCudaError(cudaError_t error, const std::string& operation);
        void profileGpuOperation(const std::string& operation_name,
                               std::function<void()> operation);
        
        double measureGpuMemoryBandwidth();
        double measureGpuComputeThroughput();
        void warmupGpuKernels();
        
        // Data conversion utilities
        thrust::device_vector<float> convertToGpuFloatVector(
            const std::vector<sep::connectors::MarketData>& market_data);
        
        std::vector<sep::connectors::MarketData> convertFromGpuFloatVector(
            const thrust::device_vector<float>& gpu_data);
        
        void transferDataToGpu(const std::vector<float>& host_data,
                              thrust::device_vector<float>& device_data,
                              cudaStream_t stream = 0);
        
        void transferDataFromGpu(const thrust::device_vector<float>& device_data,
                                std::vector<float>& host_data,
                                cudaStream_t stream = 0);
        
        // Configuration and state
        CudaWalkForwardConfig config_;
        
        // CUDA resources
        cudaDeviceProp cuda_device_props_;
        cublasHandle_t cublas_handle_;
        curandGenerator_t curand_generator_;
        
        // CUDA streams and memory
        std::vector<cudaStream_t> cuda_streams_;
        std::vector<cudaEvent_t> cuda_events_;
        void* gpu_memory_pool_ = nullptr;
        size_t gpu_memory_pool_size_ = 0;
        
        // Device memory buffers
        thrust::device_vector<float> device_input_buffer_;
        thrust::device_vector<float> device_output_buffer_;
        thrust::device_vector<float> device_temp_buffer_;
        thrust::device_vector<float> device_volume_buffer_;
        
        // Performance profiling
        mutable std::mutex profiling_mutex_;
        std::map<std::string, std::vector<float>> gpu_timing_data_;
        std::map<std::string, cudaEvent_t> profiling_start_events_;
        std::map<std::string, cudaEvent_t> profiling_end_events_;
        bool gpu_profiling_enabled_ = false;
        
        // State management
        mutable std::mutex cuda_mutex_;
        std::atomic<bool> cuda_initialized_{false};
        std::atomic<bool> streams_initialized_{false};
        std::atomic<int> active_streams_{0};
    };

    // CUDA kernel function declarations
    extern "C" {
        void cuda_walk_forward_kernel(
            const float* input_data, float* output_data,
            size_t data_size, size_t window_size, size_t num_windows,
            size_t threads_per_block, size_t num_blocks);
        
        void cuda_volumetric_sampling_kernel(
            const float* market_data, float* volume_output,
            size_t data_points, size_t depth_levels, size_t time_buckets,
            size_t threads_per_block, size_t num_blocks);
        
        void cuda_coherence_calculation_kernel(
            const float* data1, const float* data2, float* coherence_output,
            size_t data_size, size_t threads_per_block, size_t num_blocks);
        
        void cuda_tensor_operation_kernel(
            const float* input, float* output, size_t size,
            size_t threads_per_block, size_t num_blocks);
    }

    // Factory functions and utilities
    std::unique_ptr<CudaWalkForwardValidator> createCudaValidator(
        const CudaWalkForwardConfig& config = {});
    
    // CUDA system information utilities
    std::vector<std::string> enumerateCudaDevices();
    bool isCudaAvailable();
    size_t getTotalGpuMemory(int device_id = 0);
    std::string getCudaDeviceInfo(int device_id = 0);
    
    // Performance benchmarking utilities
    double benchmarkCudaPerformance(int device_id = 0);
    std::map<std::string, double> profileCudaOperations(
        const CudaWalkForwardValidator& validator);

} // namespace sep::validation::cuda