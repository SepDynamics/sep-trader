#include "metrics_calculator.h"
#include "quantum/quantum_processor.h"
#include "quantum/bitspace/qfh.h"
#include <glm/glm.hpp>
#include <vector>

namespace sep {
namespace metrics {

class MetricsCalculator::Impl {
public:
    Impl() : quantum_processor_({}) {}

    Metrics calculate_metrics(const std::vector<float>& data) {
        if (data.size() < 3) {
            return {0.0f, 0.0f, 0.0f};
        }

        std::vector<uint8_t> bitstream;
        for (float val : data) {
            uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
            for (int i = 0; i < 32; ++i) {
                bitstream.push_back((bits >> i) & 1);
            }
        }

        sep::quantum::QFHResult qfh_result = qfh_processor_.analyze(bitstream);
        
        float stability = quantum_processor_.calculateStability(qfh_result.coherence, 0.5f, 1.0f, 1.0f);

        return {qfh_result.coherence, stability, qfh_result.entropy};
    }

private:
    sep::quantum::QFHBasedProcessor qfh_processor_;
    sep::quantum::QuantumProcessor quantum_processor_;
};

MetricsCalculator::MetricsCalculator() : pimpl_(new Impl()) {}
MetricsCalculator::~MetricsCalculator() { delete pimpl_; }

Metrics MetricsCalculator::calculate_metrics(const std::vector<float>& data) {
    return pimpl_->calculate_metrics(data);
}

} // namespace metrics
} // namespace sep