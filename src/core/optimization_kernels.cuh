// SEP CUDA Optimization Kernels Header

#ifndef OPTIMIZATION_KERNELS_CUH
#define OPTIMIZATION_KERNELS_CUH

#ifdef __cplusplus
extern "C" {
#endif

void launch_optimization_kernel(float* parameters, float* gradients, int size);

#ifdef __cplusplus
}
#endif

#endif // OPTIMIZATION_KERNELS_CUH
