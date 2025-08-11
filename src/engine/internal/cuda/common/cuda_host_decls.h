#ifndef SEP_CUDA_HOST_DECLS_H
#define SEP_CUDA_HOST_DECLS_H

// Forward declarations for CUDA types used in host code
#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0
} cudaError_t;

const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif // SEP_CUDA_HOST_DECLS_H