#ifndef METRICS_CALCULATOR_H
#define METRICS_CALCULATOR_H

#include <vector>
#include <cstdint>

namespace sep {
namespace metrics {

struct Metrics {
    float coherence;
    float stability;
    float entropy;
};

class MetricsCalculator {
public:
    MetricsCalculator();
    ~MetricsCalculator();

    Metrics calculate_metrics(const std::vector<float>& data);

private:
    class Impl;
    Impl* pimpl_;
};

} // namespace metrics
} // namespace sep

#endif // METRICS_CALCULATOR_H