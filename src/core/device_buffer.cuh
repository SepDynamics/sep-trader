#pragma once

#include "core/cuda_error.cuh"
#include "cuda_includes.cuh"
#include <type_traits>
#include <stdexcept>
#include <cstdio>

namespace sep::cuda::memory {

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : data_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
        }
    }

    // Move constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        free();
    }

    // Copy from host to device
    void copyFromHost(const T* host_data, size_t count, cudaStream_t stream = 0) {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            data_,
            host_data,
            count * sizeof(T),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    // Copy from device to host
    void copyToHost(T* host_data, size_t count, cudaStream_t stream = 0) const {
        if (count > size_) {
            throw std::out_of_range("Copy count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            host_data,
            data_,
            count * sizeof(T),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }

    // Copy from another device buffer
    void copyFromDevice(const DeviceBuffer& other, size_t count, cudaStream_t stream = 0) {
        if (count > size_ || count > other.size_) {
            throw std::out_of_range("Copy count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            data_,
            other.data_,
            count * sizeof(T),
            cudaMemcpyDeviceToDevice,
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
            cudaError_t err = cudaFree(data_);
            if (err != cudaSuccess) {
                // Log error but don't throw from destructor
                char buffer[1024];
                const char* errStr = cudaGetErrorString(err);
                fprintf(stderr, "DeviceBuffer free failed: %s\n", errStr);
            }
            data_ = nullptr;
            size_ = 0;
        }
    }

    // Prevent copying
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

} // namespace sep::cuda::memory