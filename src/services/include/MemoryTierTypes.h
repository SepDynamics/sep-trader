#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <cstdint>

namespace sep {
namespace services {

/**
 * Memory tier levels in the SEP Engine
 */
enum class MemoryTierLevel {
    Tier0_UltraFast, // Ultra-fast access, extremely limited capacity
    Tier1_Fast,      // Fast access, limited capacity
    Tier2_Balanced,  // Balanced access and capacity
    Tier3_Capacity,  // High capacity, slower access
    Tier4_Archive    // Archival storage, slowest access
};

/**
 * Memory block allocation status
 */
enum class AllocationStatus {
    Succeeded,
    Failed_OutOfMemory,
    Failed_InvalidSize,
    Failed_TierFull,
    Failed_SystemError
};

/**
 * Memory block metadata
 */
struct MemoryBlockMetadata {
    std::string blockId;
    uint64_t size;
    MemoryTierLevel tier;
    uint64_t creationTime;
    uint64_t lastAccessTime;
    uint32_t accessCount;
    float priority;
    std::string contentType;
    std::map<std::string, std::string> tags;
    
    MemoryBlockMetadata() : 
        size(0), 
        tier(MemoryTierLevel::Tier2_Balanced),
        creationTime(0),
        lastAccessTime(0),
        accessCount(0),
        priority(0.0f) {}
};

/**
 * Memory block data
 */
struct MemoryBlock {
    MemoryBlockMetadata metadata;
    std::vector<uint8_t> data;
    
    MemoryBlock() {}
};

/**
 * Tier statistics
 */
struct TierStatistics {
    MemoryTierLevel tier;
    uint64_t totalCapacity;
    uint64_t usedCapacity;
    uint64_t freeCapacity;
    uint32_t blockCount;
    float averageAccessTime;
    float hitRatio;
    
    TierStatistics() : 
        tier(MemoryTierLevel::Tier2_Balanced),
        totalCapacity(0),
        usedCapacity(0),
        freeCapacity(0),
        blockCount(0),
        averageAccessTime(0.0f),
        hitRatio(0.0f) {}
};

/**
 * Memory access pattern
 */
struct MemoryAccessPattern {
    std::vector<std::string> blockSequence;
    uint32_t frequency;
    float predictability;
    uint64_t firstObservedTime;
    uint64_t lastObservedTime;
    
    MemoryAccessPattern() : 
        frequency(0),
        predictability(0.0f),
        firstObservedTime(0),
        lastObservedTime(0) {}
};

/**
 * Memory tier transition record
 */
struct TierTransitionRecord {
    std::string blockId;
    MemoryTierLevel sourceTier;
    MemoryTierLevel destinationTier;
    uint64_t transitionTime;
    std::string reason;
    
    TierTransitionRecord() : 
        sourceTier(MemoryTierLevel::Tier2_Balanced),
        destinationTier(MemoryTierLevel::Tier2_Balanced),
        transitionTime(0) {}
};

} // namespace services
} // namespace sep