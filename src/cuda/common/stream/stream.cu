#include "stream.h"
#include <iostream>

namespace sep {
namespace cuda {

// This file serves as an implementation for any non-inline functions
// defined in stream.h. Most Stream methods are inline, so this file is minimal.

// Additional Stream-related functionality could be added here
// For example, utility functions that work with streams.

// Initialize and get a pool of CUDA streams for parallel execution
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