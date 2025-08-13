#ifndef SEP_QUANTUM_QBSA_H
#define SEP_QUANTUM_QBSA_H

#include <cstdint>
#include <vector>

#include "standard_includes.h"
#include "gpu_context.h"
#include "qbsa.cuh"

namespace sep::quantum::bitspace {

    // CUDA kernel functions are declared in qbsa.cuh

    struct QBSAResult
    {
        std::vector<uint32_t> corrections;
        float correction_ratio{0.0f};
        bool collapse_detected{false};
        double damped_value{0.0};
    };

struct QBSAOptions {
    float collapse_threshold{0.6f};
};

class QBSAProcessor {
public:
    explicit QBSAProcessor(const ::sep::quantum::bitspace::QBSAOptions& options = {});
    
    // Virtual destructor for proper cleanup of derived classes
    virtual ~QBSAProcessor() = default;

    // Analyze probe indices against expected values
    virtual ::sep::quantum::bitspace::QBSAResult analyze(const std::vector<uint32_t>& probe_indices,
                               const std::vector<uint32_t>& expectations);

    // Detect collapse based on correction ratio
    virtual bool detectCollapse(const ::sep::quantum::bitspace::QBSAResult& result,
                        std::size_t total_bits) const;

    // Get options
    const ::sep::quantum::bitspace::QBSAOptions& getOptions() const;

    // Duplication logic for flux analysis
    virtual ::sep::quantum::bitspace::QBSAResult duplicateForPackage(const std::vector<uint8_t>& bitstream);

private:
    ::sep::quantum::bitspace::QBSAOptions options_;
};

} // namespace sep::quantum::bitspace

#endif // SEP_QUANTUM_QBSA_H