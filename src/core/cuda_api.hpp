#pragma once

#ifndef SEP_USE_CUDA
#error "SEP_USE_CUDA must be defined; CUDA support required."
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

