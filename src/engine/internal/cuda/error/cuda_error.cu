#include "cuda_error.cuh"

namespace sep::cuda::error {

namespace {
thread_local char error_buffer[1024];
}

CudaError::CudaError(cudaError_t error, const char* file, int line)
    : std::runtime_error(createErrorMessage(error, file, line)) {}

const char* CudaError::createErrorMessage(cudaError_t error, const char* file, int line) {
    const char* errorStr = cudaGetErrorString(error);
    snprintf(error_buffer, sizeof(error_buffer), 
             "CUDA error at %s:%d: %s (code %d)", 
             file, line, errorStr, error);
    return error_buffer;
}

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw CudaError(err, file, line);
    }
}

} // namespace sep::cuda::error