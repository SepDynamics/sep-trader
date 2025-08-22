#pragma once

#include "IMemoryTierService.h"
#include "app/ServiceBase.h"
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <string>

namespace sep {
namespace services {

/**
 * Memory Tier Service implementation
 * Provides an abstraction layer for memory tier management operations
 */
class MemoryTierService : public ServiceBase, public IMemoryTierService {
public:
    /**
     * Constructor
     */
    MemoryTierService();

    /**
     * Destructor
     */
    ~MemoryTierService() override;

    // IMemoryTierService implementation
    Result<sep::memory::MemoryBlock*> allocate(std::size_t size, sep::memory::MemoryTierEnum tier) override;
    Result<void> deallocate(sep::memory::MemoryBlock* block) override;
    Result<sep::memory::MemoryBlock*> findBlockByPtr(void* ptr) override;
    Result<sep::memory::MemoryTier*> getTier(sep::memory::MemoryTierEnum tier) override;
    Result<float> getTierUtilization(sep::memory::MemoryTierEnum tier) override;
    Result<float> getTierFragmentation(sep::memory::MemoryTierEnum tier) override;
    Result<float> getTotalUtilization() override;
    Result<float> getTotalFragmentation() override;
    Result<void> defragmentTier(sep::memory::MemoryTierEnum tier) override;
    Result<void> optimizeBlocks() override;
    Result<void> optimizeTiers() override;
    Result<sep::memory::MemoryBlock*> promoteBlock(sep::memory::MemoryBlock* block) override;
    Result<sep::memory::MemoryBlock*> demoteBlock(sep::memory::MemoryBlock* block) override;
    Result<sep::memory::MemoryBlock*> updateBlockMetrics(
        sep::memory::MemoryBlock* block, 
        float coherence, 
        float stability, 
        uint32_t generation, 
        float contextScore) override;
    Result<std::string> getMemoryAnalytics() override;
    Result<std::string> getMemoryVisualization() override;
    Result<void> configureTierPolicies(const sep::memory::MemoryThresholdConfig& config) override;
    Result<void> optimizeRedisIntegration(int optimizationLevel) override;

    Result<std::string> allocateBlock(uint64_t size, MemoryTierLevel tier, const std::string& contentType, const std::vector<uint8_t>& initialData = {}, const std::map<std::string, std::string>& tags = {}) override;
    Result<void> deallocateBlock(const std::string& blockId) override;
    Result<void> storeData(const std::string& blockId, const std::vector<uint8_t>& data, uint64_t offset = 0) override;
    Result<std::vector<uint8_t>> retrieveData(const std::string& blockId, uint64_t size, uint64_t offset = 0) override;
    Result<MemoryBlockMetadata> getBlockMetadata(const std::string& blockId) override;
    Result<void> moveBlockToTier(const std::string& blockId, MemoryTierLevel destinationTier, const std::string& reason = "Manual transition") override;
    Result<TierStatistics> getTierStatistics(MemoryTierLevel tier) override;
    Result<std::map<MemoryTierLevel, TierStatistics>> getAllTierStatistics() override;
    Result<void> configureTier(MemoryTierLevel tier, uint64_t totalCapacity, const std::map<std::string, std::string>& policies = {}) override;
    Result<void> optimizeTiers(bool aggressive = false) override;
    int registerTransitionCallback(std::function<void(const TierTransitionRecord&)> callback) override;
    Result<void> unregisterTransitionCallback(int subscriptionId) override;
    Result<std::vector<TierTransitionRecord>> getTransitionHistory(int maxRecords = 100) override;
    Result<std::vector<MemoryAccessPattern>> getAccessPatterns(uint32_t minFrequency = 5) override;


protected:
    /**
     * Initialize the service
     * @return Result<void> Success or error
     */
    Result<void> onInitialize() override;

    /**
     * Shutdown the service
     * @return Result<void> Success or error
     */
    Result<void> onShutdown() override;

private:
    // Internal implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace services
} // namespace sep