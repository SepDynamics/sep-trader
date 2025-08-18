#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>
#include <cstdint>

namespace sep {
namespace cuda {

/**
 * RAII wrapper for CUDA streams
 */
class StreamRAII {
public:
    explicit StreamRAII(unsigned int flags = cudaStreamDefault);
    ~StreamRAII() noexcept;
    
    // Move constructor and assignment
    StreamRAII(StreamRAII&& other) noexcept;
    StreamRAII& operator=(StreamRAII&& other) noexcept;
    
    // No copy operations
    StreamRAII(const StreamRAII&) = delete;
    StreamRAII& operator=(const StreamRAII&) = delete;
    
    // Get the underlying CUDA stream
    cudaStream_t get() const;
    
    // Check if stream is valid
    bool valid() const;
    
    // Synchronize the stream
    void synchronize() const;
    
private:
    cudaStream_t stream_;
};

/**
 * RAII wrapper for CUDA device memory buffers
 */
template <typename T>
class DeviceBufferRAII {
public:
    explicit DeviceBufferRAII(std::size_t count);
    ~DeviceBufferRAII() noexcept;
    
    // Move constructor and assignment
    DeviceBufferRAII(DeviceBufferRAII&& other) noexcept;
    DeviceBufferRAII& operator=(DeviceBufferRAII&& other) noexcept;
    
    // No copy operations
    DeviceBufferRAII(const DeviceBufferRAII&) = delete;
    DeviceBufferRAII& operator=(const DeviceBufferRAII&) = delete;
    
    // Get the device pointer
    T* get() const;
    
    // Get the number of elements
    std::size_t count() const;
    
    // Check if buffer is valid
    bool valid() const;
    
private:
    T* ptr_;
    std::size_t count_;
};

} // namespace cuda


/**
 * Simple RAII wrapper for CUDA resources
 */
template <typename T>
class CudaResource {
public:
    using Deleter = std::function<void(T)>;
    
    CudaResource() : resource_(nullptr), deleter_(nullptr) {}
    
    CudaResource(T resource, Deleter deleter)
        : resource_(resource), deleter_(deleter) {}
    
    ~CudaResource() {
        if (resource_ && deleter_) {
            deleter_(resource_);
        }
    }
    
    // Move constructor
    CudaResource(CudaResource&& other) 
        : resource_(other.resource_), deleter_(std::move(other.deleter_)) {
        other.resource_ = nullptr;
    }
    
    // Move assignment
    CudaResource& operator=(CudaResource&& other) {
        if (this != &other) {
            if (resource_ && deleter_) {
                deleter_(resource_);
            }
            resource_ = other.resource_;
            deleter_ = std::move(other.deleter_);
            other.resource_ = nullptr;
        }
        return *this;
    }
    
    // No copy operations
    CudaResource(const CudaResource&) = delete;
    CudaResource& operator=(const CudaResource&) = delete;
    
    // Get the underlying resource
    T get() const { return resource_; }
    
    // Release ownership of the resource
    T release() {
        T temp = resource_;
        resource_ = nullptr;
        return temp;
    }
    
private:
    T resource_;
    Deleter deleter_;
};

// UnifiedMemory class - minimal implementation for compilation
template <typename T>
class UnifiedMemory {
public:
    UnifiedMemory(size_t count) : count_(count), ptr_(nullptr) {
        if (count > 0) {
            cudaError_t err = cudaMallocManaged(&ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate unified memory");
            }
        }
    }
    
    ~UnifiedMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    // Move constructor
    UnifiedMemory(UnifiedMemory&& other) : count_(other.count_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    // Move assignment
    UnifiedMemory& operator=(UnifiedMemory&& other) {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            count_ = other.count_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    // No copy operations
    UnifiedMemory(const UnifiedMemory&) = delete;
    UnifiedMemory& operator=(const UnifiedMemory&) = delete;
    
    // Get pointer to the memory
    T* get() const { return ptr_; }
    
    // Get the number of elements
    size_t count() const { return count_; }
    
    // Get the size in bytes
    size_t size() const { return count_ * sizeof(T); }
    
private:
    size_t count_;
    T* ptr_;
};

} // namespace sep