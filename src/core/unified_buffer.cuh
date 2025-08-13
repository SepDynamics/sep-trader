#pragma once

#include "cuda_error.cuh"
#include "cuda_includes.cuh"
#include <type_traits>
#include <stdexcept>
#include <cstdio>

namespace sep::cuda::memory {

template<typename T>
class UnifiedBuffer {
public:
    UnifiedBuffer() : data_(nullptr), size_(0) {}
    
    explicit UnifiedBuffer(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMallocManaged(&data_, count * sizeof(T)));
        }
    }

    // Move constructor
    UnifiedBuffer(UnifiedBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    UnifiedBuffer& operator=(UnifiedBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~UnifiedBuffer() {
        free();
    }

    // Prefetch to device
    void prefetchToDevice(int device = 0, cudaStream_t stream = 0) {
        if (!valid()) {
            throw std::runtime_error("Cannot prefetch invalid buffer");
        }
        CUDA_CHECK(cudaMemPrefetchAsync(data_, size_ * sizeof(T), device, stream));
    }

    // Prefetch to host
    void prefetchToHost(cudaStream_t stream = 0) {
        if (!valid()) {
            throw std::runtime_error("Cannot prefetch invalid buffer");
        }
        CUDA_CHECK(cudaMemPrefetchAsync(data_, size_ * sizeof(T), cudaCpuDeviceId, stream));
    }

    // Memory advise
    void advise(cudaMemoryAdvise advice, int device = 0) {
        if (!valid()) {
            throw std::runtime_error("Cannot set advice on invalid buffer");
        }
        CUDA_CHECK(cudaMemAdvise(data_, size_ * sizeof(T), advice, device));
    }

    // Get raw pointer
    T* get() { return data_; }
    const T* get() const { return data_; }

    // Get size
    size_t size() const { return size_; }

    // Check if buffer is valid
    bool valid() const { return data_ != nullptr; }

private:
    T* data_;
    size_t size_;

    void free() {
        if (data_) {
            cudaError_t err = cudaFree(data_);
            if (err != cudaSuccess) {
                // Log error but don't throw from destructor
                const char* errStr = cudaGetErrorString(err);
                fprintf(stderr, "UnifiedBuffer free failed: %s\n", errStr);
            }
            data_ = nullptr;
            size_ = 0;
        }
    }

    // Prevent copying
    UnifiedBuffer(const UnifiedBuffer&) = delete;
    UnifiedBuffer& operator=(const UnifiedBuffer&) = delete;
};

} // namespace sep::cuda::memory