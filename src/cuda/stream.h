#pragma once

#ifndef SRC_CUDA_STREAM_MERGED_H
#define SRC_CUDA_STREAM_MERGED_H

#include "../util/stable_headers.h"
#include <cuda_runtime.h>

#include <functional>
#include <memory>

#include "cuda_error.h"

namespace sep {
namespace cuda {

// Simple callback function type for CUDA streams
typedef void (*CudaStreamCallback)(cudaStream_t stream, cudaError_t status, void* userData);

// RAII wrapper for CUDA streams
class Stream {
public:
    // Default constructor creates a new stream
    Stream() : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    // Constructor with flags
    explicit Stream(unsigned int flags) : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    // Constructor with flags and priority
    Stream(unsigned int flags, int priority) : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, flags, priority));
    }

    // Move constructor
    Stream(Stream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    // Move assignment
    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            destroy();
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    // Destructor
    ~Stream() {
        destroy();
    }

    // Delete copy constructor and assignment
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Get the underlying CUDA stream
    cudaStream_t get() const { return stream_; }
    cudaStream_t handle() const { return stream_; } // Alias for get()
    
    // Implicit conversion to cudaStream_t
    operator cudaStream_t() const { return stream_; }

    // Synchronize the stream
    void synchronize() const {
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    // Check if all operations in the stream are completed
    bool isFinished() const {
        if (!stream_) {
            return true;
        }
        return cudaStreamQuery(stream_) == cudaSuccess;
    }

    // Wait on an event
    void waitEvent(cudaEvent_t event, unsigned int flags = 0) const {
        if (stream_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, event, flags));
        }
    }

    // Record an event
    void record(cudaEvent_t event) {
        if (stream_) {
            cudaEventRecord(event, stream_);
        }
    }

    // Add a callback to the stream
    void addCallback(CudaStreamCallback callback, void* userData = nullptr, unsigned int flags = 0) const {
        if (stream_) {
            CUDA_CHECK(cudaStreamAddCallback(stream_, callback, userData, flags));
        }
    }

    bool isValid() const {
        return stream_ != nullptr;
    }

    static std::shared_ptr<Stream> create(unsigned int flags = cudaStreamDefault) {
        return std::make_shared<Stream>(flags);
    }

private:
    void destroy() {
        if (stream_) {
            CUDA_CHECK(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_;
};

// Get the default (NULL) stream
inline cudaStream_t getDefaultStream() {
    return nullptr;
}

} // namespace cuda
} // namespace sep

#endif // SRC_CUDA_STREAM_MERGED_H
