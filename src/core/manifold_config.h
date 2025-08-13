#pragma once

#include "config.h"
#include "types.h"
#include "config.h"

namespace sep::quantum::manifold {
// Default configuration values used across the engine.
extern ::sep::memory::MemoryThresholdConfig memory;
extern ::sep::QuantumThresholdConfig quantum;
extern ::sep::config::CudaConfig cuda;
extern ::sep::config::LogConfig api;
}


