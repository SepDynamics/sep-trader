#include "quantum_processor_qfh.h"

#include "standard_includes.h"
#include "types.h"
#include "types.h"  // For MemoryTierEnum
#include "processor.h"

namespace sep::quantum {


const QFHResult& QuantumProcessorQFH::getLastQFHResult() const {
    return lastQFHResult();
}

sep::memory::MemoryTierEnum QuantumProcessorQFH::determineMemoryTier(float coherence, float stability,
                                                         uint32_t generation_count) const {
    if (coherence >= 0.9f && stability >= 0.85f && generation_count >= 100)
        return sep::memory::MemoryTierEnum::LTM;
    if (coherence >= 0.7f && generation_count >= 5)
        return sep::memory::MemoryTierEnum::MTM;
    return sep::memory::MemoryTierEnum::STM;
}

}  // namespace sep::quantum
