#ifndef SEP_CUDA_DEVICE_BUFFER_H
#define SEP_CUDA_DEVICE_BUFFER_H

#include "../../common/stable_headers.h"
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "../error/cuda_error.h"
#include "buffer.h"

namespace sep {
namespace cuda {

// Device memory buffer implementation
template <typename T>
class DeviceBuffer : public Buffer<T> {
public:
    using value_type = typename Buffer<T>::value_type;
    using pointer = typename Buffer<T>::pointer;
    using const_pointer = typename Buffer<T>::const_pointer;
    using reference = typename Buffer<T>::reference;
    using const_reference = typename Buffer<T>::const_reference;
    using size_type = typename Buffer<T>::size_type;

    // Default constructor
    DeviceBuffer() : data_(nullptr), size_(0), capacity_(0) {}

    // Constructor with size
    explicit DeviceBuffer(size_type size) : data_(nullptr), size_(0), capacity_(0) {
        resize(size);
    }

    // Constructor with host data
    DeviceBuffer(const_pointer host_data, size_type size) : data_(nullptr), size_(0), capacity_(0) {
        resize(size);
        copyFromHost(host_data, size);
    }

    // Move constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    // Move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // Destructor
    ~DeviceBuffer() override {
        free();
    }

    // Delete copy constructor and assignment
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Buffer interface implementation
    pointer data() override { return data_; }
    const_pointer data() const override { return data_; }
    size_type size() const override { return size_; }
    size_type capacity() const override { return capacity_; }

    void resize(size_type new_size) override {
        if (new_size == size_) {
            return;
        }

        if (new_size > capacity_) {
            // Need to allocate more memory
            pointer new_data = nullptr;
            if (new_size > 0) {
                CUDA_CHECK(cudaMalloc(&new_data, new_size * sizeof(T)));
            }

            // If we have existing data, copy it to the new buffer
            if (data_ != nullptr && size_ > 0) {
                size_type copy_size = std::min(size_, new_size);
                CUDA_CHECK(cudaMemcpy(new_data, data_, copy_size * sizeof(T), cudaMemcpyDeviceToDevice));
            }

            // Free old buffer and update pointers
            free();
            data_ = new_data;
            capacity_ = new_size;
        }

        size_ = new_size;
    }

    void clear() override {
        size_ = 0;
    }

    void copyToHost(pointer dst, size_type count, size_type offset = 0) const override {
        if (offset + count > size_) {
            throw std::out_of_range("Copy range exceeds buffer size");
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(dst, data_ + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    void copyFromHost(const_pointer src, size_type count, size_type offset = 0) override {
        if (offset + count > size_) {
            throw std::out_of_range("Copy range exceeds buffer size");
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(data_ + offset, src, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void copyToDevice(pointer dst, size_type count, size_type offset = 0) const override {
        if (offset + count > size_) {
            throw std::out_of_range("Copy range exceeds buffer size");
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(dst, data_ + offset, count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    void copyFromDevice(const_pointer src, size_type count, size_type offset = 0) override {
        if (offset + count > size_) {
            throw std::out_of_range("Copy range exceeds buffer size");
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(data_ + offset, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

private:
    void free() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;
    }

    pointer data_;
    size_type size_;
    size_type capacity_;
};

} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_DEVICE_BUFFER_H