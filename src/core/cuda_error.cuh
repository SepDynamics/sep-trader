#ifndef SEP_CUDA_ERROR_H
#define SEP_CUDA_ERROR_H

#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include "cuda_host_decls.h"
#endif

namespace sep {
namespace cuda {

// Primary CUDA exception class used throughout the codebase
class CudaException : public std::runtime_error {
public:
    explicit CudaException(const std::string& message, cudaError_t error = cudaSuccess)
        : std::runtime_error(createErrorMessage(message, error)), error_(error) {}
    
    cudaError_t getErrorCode() const { return error_; }

private:
    static std::string createErrorMessage(const std::string& message, cudaError_t error) {
        if (error != cudaSuccess) {
            return message + " (CUDA Error: " + cudaGetErrorString(error) + ")";
        }
        return message;
    }
    
    cudaError_t error_;
};

// Legacy error handling for compatibility
namespace error {

class CudaError : public std::runtime_error {
public:
    explicit CudaError(cudaError_t error, const char* file, int line);
    using std::runtime_error::what;

private:
    static const char* createErrorMessage(cudaError_t error, const char* file, int line);
    static thread_local char error_buffer[1024];
};

void checkCudaError(cudaError_t err, const char* file, int line);

} // namespace error

} // namespace cuda
} // namespace sep

#ifdef __CUDACC__
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            throw ::sep::cuda::CudaException("CUDA operation failed", err); \
        } \
    } while (0)

#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw ::sep::cuda::CudaException("CUDA kernel launch failed", err); \
        } \
    } while (0)
#else
#define CUDA_CHECK(expr) ((void)(expr))
#define CUDA_CHECK_LAST() ((void)0)
#endif

#endif // SEP_CUDA_ERROR_H
