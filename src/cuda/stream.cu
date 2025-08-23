// Disable fpclassify functions that cause conflicts with CUDA internal headers
#define _DISABLE_FPCLASSIFY_FUNCTIONS 1
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS 1

// CUDA Stream Implementation
#include <cuda_runtime.h>
#include <vector>

#include "cuda/stream.h"

namespace sep {
namespace cuda {

// Stream class implementation
Stream::Stream(unsigned int flags) : stream_(nullptr) {
    cudaStreamCreateWithFlags(&stream_, flags);
}

Stream::~Stream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

Stream::Stream(Stream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void Stream::synchronize() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
}

bool Stream::isValid() const {
    return stream_ != nullptr;
}

cudaStream_t Stream::handle() const {
    return stream_;
}

// Create a pool of CUDA streams for parallel execution
std::vector<Stream> createStreamPool(unsigned int num_streams) {
    std::vector<Stream> stream_pool;
    stream_pool.reserve(num_streams);
    
    for (unsigned int i = 0; i < num_streams; ++i) {
        stream_pool.emplace_back(cudaStreamNonBlocking);
    }
    
    return stream_pool;
}

// Utility functions
cudaStream_t createStream(unsigned int flags) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, flags);
    return stream;
}

void destroyStream(cudaStream_t stream) {
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

void synchronizeStream(cudaStream_t stream) {
    if (stream) {
        cudaStreamSynchronize(stream);
    }
}

bool isStreamComplete(cudaStream_t stream) {
    if (!stream) return true;
    cudaError_t result = cudaStreamQuery(stream);
    return result == cudaSuccess;
}

} // namespace cuda
} // namespace sep