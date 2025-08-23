#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include "core/quantum_types.h" // for Pattern::MemoryTier

namespace sep::memory {

struct BlockHandle {
    uint64_t addr{};
    size_t   bytes{};
    ::sep::quantum::Pattern::MemoryTier tier{::sep::quantum::Pattern::MemoryTier::Hot};
};

struct TierStats {
    size_t hot_blocks{0};
    size_t warm_blocks{0};
    size_t cold_blocks{0};
    size_t hot_bytes{0};
    size_t warm_bytes{0};
    size_t cold_bytes{0};
    size_t transitions{0};
};

class MemoryTierService {
public:
    BlockHandle alloc(size_t bytes, ::sep::quantum::Pattern::MemoryTier tier);
    void        free(BlockHandle handle);
    void        promote(BlockHandle& handle, ::sep::quantum::Pattern::MemoryTier newTier);
    void*       map(BlockHandle handle);
    TierStats   stats() const { return stats_; }

private:
    using Tier = ::sep::quantum::Pattern::MemoryTier;
    struct BlockInfo { void* ptr; size_t bytes; Tier tier; };
    std::unordered_map<uint64_t, BlockInfo> blocks_;
    TierStats stats_{};
};

} // namespace sep::memory

