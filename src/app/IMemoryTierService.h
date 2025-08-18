#pragma once

#ifndef SRC_APP_IMEMORYTIERSERVICE_MERGED_H
#define SRC_APP_IMEMORYTIERSERVICE_MERGED_H

#include "IService.h"
#include "MemoryTierTypes.h"
#include "util/result.h"
#include <array>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

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

class IMemoryTierService : public virtual IService {
public:
    virtual ~IMemoryTierService() = default;

    // Methods from src/app/include/IMemoryTierService.h
    virtual Result<std::string> allocateBlock(uint64_t size, MemoryTierLevel tier, const std::string& contentType, const std::vector<uint8_t>& initialData = {}, const std::map<std::string, std::string>& tags = {}) = 0;
    virtual Result<void> deallocateBlock(const std::string& blockId) = 0;
    virtual Result<void> storeData(const std::string& blockId, const std::vector<uint8_t>& data, uint64_t offset = 0) = 0;
    virtual Result<std::vector<uint8_t>> retrieveData(const std::string& blockId, uint64_t size, uint64_t offset = 0) = 0;
    virtual Result<MemoryBlockMetadata> getBlockMetadata(const std::string& blockId) = 0;
    virtual Result<void> moveBlockToTier(const std::string& blockId, MemoryTierLevel destinationTier, const std::string& reason = "Manual transition") = 0;
    virtual Result<TierStatistics> getTierStatistics(MemoryTierLevel tier) = 0;
    virtual Result<std::map<MemoryTierLevel, TierStatistics>> getAllTierStatistics() = 0;
    virtual Result<void> configureTier(MemoryTierLevel tier, uint64_t totalCapacity, const std::map<std::string, std::string>& policies = {}) = 0;
    virtual Result<void> optimizeTiers(bool aggressive = false) = 0;
    virtual int registerTransitionCallback(std::function<void(const TierTransitionRecord&)> callback) = 0;
    virtual Result<void> unregisterTransitionCallback(int subscriptionId) = 0;
    virtual Result<std::vector<TierTransitionRecord>> getTransitionHistory(int maxRecords = 100) = 0;
    virtual Result<std::vector<MemoryAccessPattern>> getAccessPatterns(uint32_t minFrequency = 5) = 0;

    // Methods from src/app/memory/IMemoryTierService.h
    virtual Result<sep::memory::MemoryBlock*> allocate(std::size_t size, sep::memory::MemoryTierEnum tier) = 0;
    virtual Result<void> deallocate(sep::memory::MemoryBlock* block) = 0;
    virtual Result<sep::memory::MemoryBlock*> findBlockByPtr(void* ptr) = 0;
    virtual Result<sep::memory::MemoryTier*> getTier(sep::memory::MemoryTierEnum tier) = 0;
    virtual Result<float> getTierUtilization(sep::memory::MemoryTierEnum tier) = 0;
    virtual Result<float> getTierFragmentation(sep::memory::MemoryTierEnum tier) = 0;
    virtual Result<float> getTotalUtilization() = 0;
    virtual Result<float> getTotalFragmentation() = 0;
    virtual Result<void> defragmentTier(sep::memory::MemoryTierEnum tier) = 0;
    virtual Result<void> optimizeBlocks() = 0;
    virtual Result<void> optimizeTiers() = 0;
    virtual Result<sep::memory::MemoryBlock*> promoteBlock(sep::memory::MemoryBlock* block) = 0;
    virtual Result<sep::memory::MemoryBlock*> demoteBlock(sep::memory::MemoryBlock* block) = 0;
    virtual Result<sep::memory::MemoryBlock*> updateBlockMetrics(sep::memory::MemoryBlock* block, float coherence, float stability, uint32_t generation, float contextScore) = 0;
    virtual Result<std::string> getMemoryAnalytics() = 0;
    virtual Result<std::string> getMemoryVisualization() = 0;
    virtual Result<void> configureTierPolicies(const sep::memory::MemoryThresholdConfig& config) = 0;
    virtual Result<void> optimizeRedisIntegration(int optimizationLevel) = 0;
};

} // namespace services
} // namespace sep

#endif // SRC_APP_IMEMORYTIERSERVICE_MERGED_H
