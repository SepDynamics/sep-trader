#pragma once

#include "IService.h"
#include "MemoryTierTypes.h"
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace sep {
namespace services {

/**
 * Interface for the Memory Tier Service
 * Responsible for tiered memory management, block allocation, promotion/demotion,
 * and memory access optimization
 */
class IMemoryTierService : public IService {
public:
    /**
     * Allocate a memory block in the specified tier
     * @param size Size of the block in bytes
     * @param tier Target memory tier
     * @param contentType Type of content to be stored
     * @param initialData Optional initial data to store
     * @param tags Optional metadata tags
     * @return Result containing allocated block ID or error
     */
    virtual Result<std::string> allocateBlock(
        uint64_t size,
        MemoryTierLevel tier,
        const std::string& contentType,
        const std::vector<uint8_t>& initialData = {},
        const std::map<std::string, std::string>& tags = {}) = 0;
    
    /**
     * Deallocate a memory block
     * @param blockId ID of the block to deallocate
     * @return Result<void> Success or error
     */
    virtual Result<void> deallocateBlock(const std::string& blockId) = 0;
    
    /**
     * Store data in a memory block
     * @param blockId ID of the target block
     * @param data Data to store
     * @param offset Offset within the block
     * @return Result<void> Success or error
     */
    virtual Result<void> storeData(
        const std::string& blockId,
        const std::vector<uint8_t>& data,
        uint64_t offset = 0) = 0;
    
    /**
     * Retrieve data from a memory block
     * @param blockId ID of the block to retrieve from
     * @param size Number of bytes to retrieve
     * @param offset Offset within the block
     * @return Result containing retrieved data or error
     */
    virtual Result<std::vector<uint8_t>> retrieveData(
        const std::string& blockId,
        uint64_t size,
        uint64_t offset = 0) = 0;
    
    /**
     * Get metadata for a memory block
     * @param blockId ID of the block
     * @return Result containing block metadata or error
     */
    virtual Result<MemoryBlockMetadata> getBlockMetadata(const std::string& blockId) = 0;
    
    /**
     * Move a memory block to a different tier
     * @param blockId ID of the block to move
     * @param destinationTier Destination tier
     * @param reason Reason for the tier transition
     * @return Result<void> Success or error
     */
    virtual Result<void> moveBlockToTier(
        const std::string& blockId,
        MemoryTierLevel destinationTier,
        const std::string& reason = "Manual transition") = 0;
    
    /**
     * Get statistics for a specific memory tier
     * @param tier The tier to get statistics for
     * @return Result containing tier statistics or error
     */
    virtual Result<TierStatistics> getTierStatistics(MemoryTierLevel tier) = 0;
    
    /**
     * Get statistics for all memory tiers
     * @return Result containing map of tier to statistics or error
     */
    virtual Result<std::map<MemoryTierLevel, TierStatistics>> getAllTierStatistics() = 0;
    
    /**
     * Configure a tier's capacity and policies
     * @param tier The tier to configure
     * @param totalCapacity Total capacity in bytes
     * @param policies Tier-specific policies
     * @return Result<void> Success or error
     */
    virtual Result<void> configureTier(
        MemoryTierLevel tier,
        uint64_t totalCapacity,
        const std::map<std::string, std::string>& policies = {}) = 0;
    
    /**
     * Run automatic tier optimization
     * @param aggressive If true, performs more aggressive optimization
     * @return Result<void> Success or error
     */
    virtual Result<void> optimizeTiers(bool aggressive = false) = 0;
    
    /**
     * Register a callback for tier transition events
     * @param callback Function to call when transitions occur
     * @return Subscription ID for the callback
     */
    virtual int registerTransitionCallback(
        std::function<void(const TierTransitionRecord&)> callback) = 0;
    
    /**
     * Unregister a tier transition callback
     * @param subscriptionId ID returned from registerTransitionCallback
     * @return Result<void> Success or error
     */
    virtual Result<void> unregisterTransitionCallback(int subscriptionId) = 0;
    
    /**
     * Get recent tier transition history
     * @param maxRecords Maximum number of records to retrieve
     * @return Result containing transition records or error
     */
    virtual Result<std::vector<TierTransitionRecord>> getTransitionHistory(
        int maxRecords = 100) = 0;
    
    /**
     * Get detected memory access patterns
     * @param minFrequency Minimum frequency threshold for patterns
     * @return Result containing access patterns or error
     */
    virtual Result<std::vector<MemoryAccessPattern>> getAccessPatterns(
        uint32_t minFrequency = 5) = 0;
};

} // namespace services
} // namespace sep