#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace sep {
namespace metrics_library {

struct MetricsResult {
    float coherence;
    float stability;
    float entropy;
};

// Function to calculate coherence, stability, and entropy from a bitstream
MetricsResult calculateMetrics(const std::vector<uint8_t>& bitstream);

} // namespace metrics_library
} // namespace sep
