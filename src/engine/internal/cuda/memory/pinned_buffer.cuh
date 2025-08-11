#pragma once

#include "engine/internal/cuda/error/cuda_error.cuh"
#include "engine/internal/cuda/common/cuda_includes.cuh"
#include <type_traits>
#include <stdexcept>
#include <cstdio>

namespace sep::cuda::memory {

template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer() : data_(nullptr), size_(0) {}
    
    explicit PinnedBuffer(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaHostAlloc(&data_, count * sizeof(T), cudaHostAllocDefault));
        }
    }

    // Move constructor
    PinnedBuffer(PinnedBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~PinnedBuffer() {
        free();
    }

    // Copy to device buffer
    void copyToDevice(T* device_ptr, size_t count, cudaStream_t stream = 0) const {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            device_ptr,
            data_,
            count * sizeof(T),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    // Copy from device buffer
    void copyFromDevice(const T* device_ptr, size_t count, cudaStream_t stream = 0) {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            data_,
            device_ptr,
            count * sizeof(T),
            cudaMemcpyDeviceToHost,
            stream
        ));
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
            cudaError_t err = cudaFreeHost(data_);
            if (err != cudaSuccess) {
                // Log error but don't throw from destructor
                const char* errStr = cudaGetErrorString(err);
                fprintf(stderr, "PinnedBuffer free failed: %s\n", errStr);
            }
            data_ = nullptr;
            size_ = 0;
        }
    }

    // Prevent copying
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

} // namespace sep::cuda::memory