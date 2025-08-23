#pragma once
#include <vector>
#include <cstdint>

namespace sep {
namespace engine {
class EngineFacade {
public:
    void process(const std::vector<uint64_t>& mask) { last_mask_ = mask; }
    const std::vector<uint64_t>& last_mask() const { return last_mask_; }
private:
    std::vector<uint64_t> last_mask_;
};
} // namespace engine

namespace quantum {
class QFHBasedProcessor {
public:
    double analyze(const std::vector<uint64_t>& mask) { return static_cast<double>(mask.size()); }
};
} // namespace quantum
} // namespace sep

