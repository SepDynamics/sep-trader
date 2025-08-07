#pragma once

// Only include the real CUDA headers - no mock types when CUDA SDK is available
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>

namespace sep::cuda {

}
