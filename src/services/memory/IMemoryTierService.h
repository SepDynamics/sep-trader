#pragma once

#include "../include/Result.h"
#include "../include/IService.h"
#include <cstddef>
#include <cstdint>  // Add this for uint32_t
#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace sep {
namespace memory {
    class MemoryTier;
    struct MemoryBlock;
    enum class MemoryTierEnum;
    struct MemoryThresholdConfig;
}
}

namespace sep {
namespace services {

/**
 * Memory Tier Service interface
 * Provides an abstraction layer for memory tier management operations
 */
class IMemoryTierService : public virtual IService {
public:
    virtual ~IMemoryTierService() = default;

    /**
     * Allocate memory block in specified tier
     * @param size Size of the memory block to allocate
     * @param tier Target memory tier
     * @return Result containing pointer to allocated memory block or error
     */
    virtual Result<sep::memory::MemoryBlock*> allocate(std::size_t size, sep::memory::MemoryTierEnum tier) = 0;

    /**
     * Deallocate memory block
     * @param block Memory block to deallocate
     * @return Result indicating success or error
     */
    virtual Result<void> deallocate(sep::memory::MemoryBlock* block) = 0;

    /**
     * Find memory block by pointer
     * @param ptr Pointer to memory
     * @return Result containing pointer to memory block or error
     */
    virtual Result<sep::memory::MemoryBlock*> findBlockByPtr(void* ptr) = 0;

    /**
     * Get memory tier by enum
     * @param tier Memory tier enum
     * @return Result containing pointer to memory tier or error
     */
    virtual Result<sep::memory::MemoryTier*> getTier(sep::memory::MemoryTierEnum tier) = 0;

    /**
     * Get tier utilization (0.0 - 1.0)
     * @param tier Memory tier enum
     * @return Result containing utilization or error
     */
    virtual Result<float> getTierUtilization(sep::memory::MemoryTierEnum tier) = 0;

    /**
     * Get tier fragmentation (0.0 - 1.0)
     * @param tier Memory tier enum
     * @return Result containing fragmentation or error
     */
    virtual Result<float> getTierFragmentation(sep::memory::MemoryTierEnum tier) = 0;

    /**
     * Get total memory utilization across all tiers
     * @return Result containing total utilization or error
     */
    virtual Result<float> getTotalUtilization() = 0;

    /**
     * Get total memory fragmentation across all tiers
     * @return Result containing total fragmentation or error
     */
    virtual Result<float> getTotalFragmentation() = 0;

    /**
     * Defragment specified memory tier
     * @param tier Memory tier enum
     * @return Result indicating success or error
     */
    virtual Result<void> defragmentTier(sep::memory::MemoryTierEnum tier) = 0;

    /**
     * Optimize block metrics and layout across all tiers
     * @return Result indicating success or error
     */
    virtual Result<void> optimizeBlocks() = 0;

    /**
     * Optimize tiers (defragment if necessary)
     * @return Result indicating success or error
     */
    virtual Result<void> optimizeTiers() = 0;

    /**
     * Promote memory block to next tier
     * @param block Memory block to promote
     * @return Result containing promoted block or error
     */
    virtual Result<sep::memory::MemoryBlock*> promoteBlock(sep::memory::MemoryBlock* block) = 0;

    /**
     * Demote memory block to previous tier
     * @param block Memory block to demote
     * @return Result containing demoted block or error
     */
    virtual Result<sep::memory::MemoryBlock*> demoteBlock(sep::memory::MemoryBlock* block) = 0;

    /**
     * Update block metrics
     * @param block Memory block to update
     * @param coherence Coherence value
     * @param stability Stability value
     * @param generation Generation count
     * @param contextScore Context relevance score
     * @return Result containing updated block or error
     */
    virtual Result<sep::memory::MemoryBlock*> updateBlockMetrics(
        sep::memory::MemoryBlock* block, 
        float coherence, 
        float stability, 
        uint32_t generation, 
        float contextScore) = 0;

    /**
     * Get memory usage analytics
     * @return Result containing memory analytics data or error
     */
    virtual Result<std::string> getMemoryAnalytics() = 0;

    /**
     * Get memory tier status for visualization
     * @return Result containing tier visualization data or error
     */
    virtual Result<std::string> getMemoryVisualization() = 0;

    /**
     * Configure tier management policies
     * @param config Memory threshold configuration
     * @return Result indicating success or error
     */
    virtual Result<void> configureTierPolicies(const sep::memory::MemoryThresholdConfig& config) = 0;

    /**
     * Optimize Redis integration for pattern persistence
     * @param optimizationLevel Optimization level (0-3)
     * @return Result indicating success or error
     */
    virtual Result<void> optimizeRedisIntegration(int optimizationLevel) = 0;
};

} // namespace services
} // namespace sep