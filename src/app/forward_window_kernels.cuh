#ifndef SEP_APPS_OANDA_TRADER_FORWARD_WINDOW_KERNELS_CUH
#define SEP_APPS_OANDA_TRADER_FORWARD_WINDOW_KERNELS_CUH

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "core/trajectory.h"
#include "core/forward_window_result.h"

namespace sep::apps::cuda {

// Use full namespace paths to avoid CUDA compilation issues
// using TrajectoryPoint = ::sep::core::TrajectoryPoint;
// using DampedValue = ::sep::core::DampedValue;
// using ForwardWindowResult = ::sep::core::ForwardWindowResult;

// Device-side equivalent of TrajectoryPoint
struct TrajectoryPointDevice {
    double value;
    uint64_t timestamp;
};

// Device-side equivalent of DampedValue
struct DampedValueDevice {
    double final_value;
    double confidence;
    bool converged;
};

// Launcher for the trajectory analysis kernel
void launchTrajectoryKernel(const TrajectoryPointDevice* trajectory_points,
                            DampedValueDevice* results,
                            int num_trajectories,
                            int trajectory_length,
                            cudaStream_t stream);

// CPU version for testing and fallback
void simulateForwardWindowMetrics(const std::vector<::sep::core::TrajectoryPoint>& trajectories, std::vector<::sep::core::DampedValue>& results);

} // namespace sep::apps::cuda

#endif // SEP_APPS_OANDA_TRADER_FORWARD_WINDOW_KERNELS_CUH
