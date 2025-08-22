#pragma once

#include "core/config.h"
#include "core/types.h"

namespace sep::quantum::manifold {
// Default configuration values used across the engine.
extern ::sep::memory::MemoryThresholdConfig memory;
extern ::sep::QuantumThresholdConfig quantum;
extern ::sep::config::CudaConfig cuda;
extern ::sep::config::LogConfig api;
}


