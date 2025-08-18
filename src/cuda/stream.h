#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace sep {
namespace cuda {

/**
 * RAII wrapper for CUDA streams
 */
class Stream {
public:
    explicit Stream(unsigned int flags = cudaStreamDefault);
    ~Stream();
    
    // Move constructor and assignment
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    
    // No copy operations
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    
    // Stream operations
    void synchronize();
    bool isValid() const;
    cudaStream_t handle() const;
    
private:
    cudaStream_t stream_;
};

/**
 * Stream pool management
 */
std::vector<Stream> createStreamPool(unsigned int num_streams);

/**
 * Utility functions
 */
cudaStream_t createStream(unsigned int flags = cudaStreamDefault);
void destroyStream(cudaStream_t stream);
void synchronizeStream(cudaStream_t stream);
bool isStreamComplete(cudaStream_t stream);

} // namespace cuda
} // namespace sep