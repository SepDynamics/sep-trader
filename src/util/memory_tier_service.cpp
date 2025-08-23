#include "util/memory_tier_service.h"

#include <cstring>

namespace sep::memory {

using Tier = ::sep::quantum::Pattern::MemoryTier;

BlockHandle MemoryTierService::alloc(size_t bytes, Tier tier) {
    void* ptr = ::operator new(bytes);
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    blocks_[addr] = {ptr, bytes, tier};
    BlockHandle handle{addr, bytes, tier};
    switch (tier) {
        case Tier::Hot:  stats_.hot_blocks++;  stats_.hot_bytes += bytes;  break;
        case Tier::Warm: stats_.warm_blocks++; stats_.warm_bytes += bytes; break;
        case Tier::Cold: stats_.cold_blocks++; stats_.cold_bytes += bytes; break;
    }
    return handle;
}

void MemoryTierService::free(BlockHandle handle) {
    auto it = blocks_.find(handle.addr);
    if (it == blocks_.end()) return;
    auto info = it->second;
    switch (info.tier) {
        case Tier::Hot:  stats_.hot_blocks--;  stats_.hot_bytes -= info.bytes;  break;
        case Tier::Warm: stats_.warm_blocks--; stats_.warm_bytes -= info.bytes; break;
        case Tier::Cold: stats_.cold_blocks--; stats_.cold_bytes -= info.bytes; break;
    }
    ::operator delete(info.ptr);
    blocks_.erase(it);
}

void MemoryTierService::promote(BlockHandle& handle, Tier newTier) {
    auto it = blocks_.find(handle.addr);
    if (it == blocks_.end()) throw std::invalid_argument("invalid handle");
    if (it->second.tier == newTier) throw std::invalid_argument("invalid transition");
    // Update stats
    switch (it->second.tier) {
        case Tier::Hot:  stats_.hot_blocks--;  stats_.hot_bytes -= it->second.bytes; break;
        case Tier::Warm: stats_.warm_blocks--; stats_.warm_bytes -= it->second.bytes; break;
        case Tier::Cold: stats_.cold_blocks--; stats_.cold_bytes -= it->second.bytes; break;
    }
    switch (newTier) {
        case Tier::Hot:  stats_.hot_blocks++;  stats_.hot_bytes += it->second.bytes; break;
        case Tier::Warm: stats_.warm_blocks++; stats_.warm_bytes += it->second.bytes; break;
        case Tier::Cold: stats_.cold_blocks++; stats_.cold_bytes += it->second.bytes; break;
    }
    stats_.transitions++;
    it->second.tier = newTier;
    handle.tier = newTier;
}

void* MemoryTierService::map(BlockHandle handle) {
    auto it = blocks_.find(handle.addr);
    if (it == blocks_.end()) return nullptr;
    return it->second.ptr;
}

} // namespace sep::memory

