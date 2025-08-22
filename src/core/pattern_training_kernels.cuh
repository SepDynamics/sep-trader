// SEP CUDA Pattern Training Kernels Header
// GPU-accelerated pattern recognition training

#ifndef PATTERN_TRAINING_KERNELS_CUH
#define PATTERN_TRAINING_KERNELS_CUH

#ifdef __cplusplus
extern "C" {
#endif

void launch_pattern_training_kernel(float* input, float* output, int size);

#ifdef __cplusplus
}
#endif

#endif // PATTERN_TRAINING_KERNELS_CUH
