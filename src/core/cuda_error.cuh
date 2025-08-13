#ifndef SEP_CUDA_ERROR_H
#define SEP_CUDA_ERROR_H

#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include "cuda_host_decls.h"
#endif

namespace sep::cuda::error {

class CudaError : public std::runtime_error {
public:
    explicit CudaError(cudaError_t error, const char* file, int line);
    using std::runtime_error::what;

private:
    static const char* createErrorMessage(cudaError_t error, const char* file, int line);
    static thread_local char error_buffer[1024];
};

void checkCudaError(cudaError_t err, const char* file, int line);

} // namespace sep::cuda::error

#ifdef __CUDACC__
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        ::sep::cuda::error::checkCudaError(err, __FILE__, __LINE__); \
    } while (0)
#else
#define CUDA_CHECK(expr) ((void)(expr))
#endif

#endif // SEP_CUDA_ERROR_H