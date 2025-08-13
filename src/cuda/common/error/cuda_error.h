#ifndef SEP_CUDA_ERROR_H
#define SEP_CUDA_ERROR_H

#include "../../common/stable_headers.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace sep {
namespace cuda {

// Custom exception class for CUDA errors
class CudaException : public std::runtime_error {
public:
    explicit CudaException(const std::string& message, cudaError_t error_code = cudaSuccess)
        : std::runtime_error(message + ": " + cudaGetErrorString(error_code))
        , error_code_(error_code) {}

    cudaError_t getErrorCode() const { return error_code_; }

private:
    cudaError_t error_code_;
};

// Check CUDA error and throw exception if not successful
inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw CudaException(
            std::string("CUDA error at ") + file + ":" + std::to_string(line),
            error
        );
    }
}

// Convenience macro for error checking
#define CUDA_CHECK(expr) checkCudaError((expr), __FILE__, __LINE__)

// Check for asynchronous errors
inline void checkLastCudaError(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException(
            std::string("CUDA error at ") + file + ":" + std::to_string(line),
            error
        );
    }
}

// Convenience macro for checking last error
#define CUDA_CHECK_LAST() checkLastCudaError(__FILE__, __LINE__)

} // namespace cuda
} // namespace sep

#endif // SEP_CUDA_ERROR_H