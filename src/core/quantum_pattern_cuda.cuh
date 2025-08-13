// SEP CUDA Quantum Pattern Processing Header

#ifndef QUANTUM_PATTERN_CUDA_CUH
#define QUANTUM_PATTERN_CUDA_CUH

#ifdef __cplusplus
extern "C" {
#endif

void launch_quantum_pattern_kernel(float* data, int size);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_PATTERN_CUDA_CUH
