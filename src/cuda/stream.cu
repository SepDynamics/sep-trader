// CUDA Stream Implementation
#include <cuda_runtime.h>
#include <vector>

#include "stream.h"

namespace sep {
namespace cuda {

// Create a pool of CUDA streams for parallel execution
std::vector<Stream> createStreamPool(unsigned int num_streams) {
    std::vector<Stream> stream_pool;
    stream_pool.reserve(num_streams);
    
    for (unsigned int i = 0; i < num_streams; ++i) {
        stream_pool.emplace_back(cudaStreamNonBlocking);
    }
    
    return stream_pool;
}

} // namespace cuda
} // namespace sep