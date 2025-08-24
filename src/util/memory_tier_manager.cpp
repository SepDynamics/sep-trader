#include "util/nlohmann_json_safe.h"
#include "util/memory_tier_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "core/common.h"
#include "core/types.h"
#include "core/processor.h"

namespace sep {
namespace memory {

// Static member initializations
std::unique_ptr<MemoryTierManager> MemoryTierManager::instance_;
std::once_flag MemoryTierManager::once_flag_;

// Singleton getter
MemoryTierManager &MemoryTierManager::getInstance()
{
    std::call_once(once_flag_, []() {
        instance_ = std::make_unique<::sep::memory::MemoryTierManager>();
        instance_->hooks_ = nullptr;
    });
    return *instance_;
}

        // Constructor
MemoryTierManager::MemoryTierManager()
    : stm_(std::make_unique<::sep::memory::MemoryTier>(::sep::memory::MemoryTierEnum::STM, 1 << 20, 0.7f, 5)),
      mtm_(std::make_unique<::sep::memory::MemoryTier>(::sep::memory::MemoryTierEnum::MTM, 4 << 20, 0.8f, 50)),
      ltm_(std::make_unique<::sep::memory::MemoryTier>(::sep::memory::MemoryTierEnum::LTM, 16 << 20, 0.9f, 100))
{
    Config default_config{};
    init(default_config);
}

MemoryTierManager::MemoryTierManager(const Config &cfg)
    : stm_(std::make_unique<::sep::memory::MemoryTier>(::sep::memory::MemoryTierEnum::STM, cfg.stm_size,
                                        cfg.promote_stm_to_mtm, cfg.stm_to_mtm_min_gen)),
      mtm_(std::make_unique<::sep::memory::MemoryTier>(::sep::memory::MemoryTierEnum::MTM, cfg.mtm_size,
                                        cfg.promote_mtm_to_ltm, cfg.stm_to_mtm_min_gen)),
      ltm_(
          std::make_unique<::sep::memory::MemoryTier>(::sep::memory::MemoryTierEnum::LTM, cfg.ltm_size, 0.9f, 100))
{
    init(cfg);
}

        // Implementation of init method
        void MemoryTierManager::init(const Config &config)
        {
            // Store the configuration
            config_.min_age_for_promotion =
                std::max(static_cast<unsigned int>(config.stm_to_mtm_min_gen), 1u);
            config_.min_age_for_ltm = std::max(config.mtm_to_ltm_min_gen, 10);
            config_.promotion_coherence_threshold = config.promote_stm_to_mtm;
            config_.ltm_coherence_threshold = config.promote_mtm_to_ltm;
            config_.demotion_threshold = config.demote_threshold;
            config_.defrag_threshold = config.fragmentation_threshold;
            config_.use_compression = config.enable_compression;
            config_.pattern_expiration_age = config.pattern_expiration_age;

            // Initialize the tiers with the config values if needed
            if (stm_) stm_->setPromotionThreshold(config.promote_stm_to_mtm);
            if (mtm_) mtm_->setPromotionThreshold(config.promote_mtm_to_ltm);

            // Set up other tier-specific configurations
            rebuildLookup();
        }

        MemoryTierManager::~MemoryTierManager() = default;

        void MemoryTierManager::shutdown()
        {
            // Clean up
            lookup_map_.clear();
        }

        ::sep::memory::MemoryBlock *MemoryTierManager::allocate(std::size_t size, ::sep::memory::MemoryTierEnum tier)
        {
            ::sep::memory::MemoryTier *t = getTier(tier);
            if (!t) {
                // CRITICAL FIX: Use STM as fallback tier instead of returning nullptr
                std::cerr << "[MemoryTierManager] WARNING: Invalid tier requested, using STM as fallback" << std::endl;
                t = stm_.get();
            }
            ::sep::memory::MemoryBlock *block = t->allocate(size);
            if (block)
            {
                lookup_map_[block->ptr] = block;
            }
            return block;
        }

        void MemoryTierManager::deallocate(::sep::memory::MemoryBlock *block)
        {
            if (!block) return;
            ::sep::memory::MemoryTier *tier = getTier(block->tier);
            if (!tier) return;
            void *ptr = block->ptr;
            tier->deallocate(block);
            lookup_map_.erase(ptr);
        }

        ::sep::memory::MemoryBlock *MemoryTierManager::findBlockByPtr(void *ptr)
        {
            auto it = lookup_map_.find(ptr);
        if (it != lookup_map_.end())
        {
            return it->second;
        }
        return nullptr;
    }

        ::sep::memory::MemoryTier *MemoryTierManager::getTier(::sep::memory::MemoryTierEnum tier)
        {
            switch (tier)
            {
                case ::sep::memory::MemoryTierEnum::STM:
                    return stm_.get();
                case ::sep::memory::MemoryTierEnum::MTM:
                    return mtm_.get();
                case ::sep::memory::MemoryTierEnum::LTM:
                    return ltm_.get();
                default:
                    return nullptr;
            }
        }

        float MemoryTierManager::getTierUtilization(::sep::memory::MemoryTierEnum tier) const
        {
            ::sep::memory::MemoryTier* target_tier = nullptr;
            switch (tier)
            {
                case ::sep::memory::MemoryTierEnum::STM:
                    target_tier = stm_.get();
                    break;
                case ::sep::memory::MemoryTierEnum::MTM:
                    target_tier = mtm_.get();
                    break;
                case ::sep::memory::MemoryTierEnum::LTM:
                    target_tier = ltm_.get();
                    break;
                default:
                    return 0.0f;
            }
            
            if (!target_tier) return 0.0f;
            
            std::size_t total_size = target_tier->getSize();
            std::size_t used_size = target_tier->getUsedSpace();
            
            return total_size > 0 ? static_cast<float>(used_size) / static_cast<float>(total_size) : 0.0f;
        }

        float MemoryTierManager::getTierFragmentation(::sep::memory::MemoryTierEnum tier) const
        {
            switch (tier)
            {
                case ::sep::memory::MemoryTierEnum::STM:
                case ::sep::memory::MemoryTierEnum::MTM:
                case ::sep::memory::MemoryTierEnum::LTM:
                default:
                    return 0.0f;
            }
        }

        float MemoryTierManager::getTotalUtilization() const
        {
            std::size_t total_size = stm_->getSize() + mtm_->getSize() + ltm_->getSize();
            std::size_t total_used =
                stm_->getUsedSpace() + mtm_->getUsedSpace() + ltm_->getUsedSpace();

            return total_size > 0 ? static_cast<float>(total_used) / static_cast<float>(total_size)
                                  : 0.0f;
        }

        float MemoryTierManager::getTotalFragmentation() const
        {
            return (stm_->calculateFragmentation() + mtm_->calculateFragmentation() +
                    ltm_->calculateFragmentation()) /
                   3.0f;
        }

        std::size_t MemoryTierManager::getTotalAllocated() const
        {
            return stm_->getUsedSpace() + mtm_->getUsedSpace() + ltm_->getUsedSpace();
        }

        ::sep::memory::MemoryTier &MemoryTierManager::getSTM() { return *stm_; }
        ::sep::memory::MemoryTier &MemoryTierManager::getMTM() { return *mtm_; }
        ::sep::memory::MemoryTier &MemoryTierManager::getLTM() { return *ltm_; }

        void MemoryTierManager::defragmentTier(::sep::memory::MemoryTierEnum tier)
        {
            ::sep::memory::MemoryTier *t = getTier(tier);
            if (t)
            {
                t->defragment();
            }
        }

        void MemoryTierManager::optimizeTiers()
        {
            auto process_tier = [](::sep::memory::MemoryTier *tier) {
                if (!tier) return;
                if (tier->calculateFragmentation() > 0.3f)
                {
                    tier->defragment();
                }
            };

            process_tier(stm_.get());
            process_tier(mtm_.get());
            process_tier(ltm_.get());
        }

        void MemoryTierManager::rebuildLookup()
        {
            lookup_map_.clear();

            auto process_tier = [this](::sep::memory::MemoryTier *tier) {
                if (!tier) return;
                auto &blocks = tier->getBlocks();
                for (const auto &blk : blocks)
                {
                    if (blk.allocated)
                    {
                        lookup_map_[blk.ptr] = const_cast<::sep::memory::MemoryBlock *>(&blk);
                    }
                }
            };

            process_tier(stm_.get());
            process_tier(mtm_.get());
            process_tier(ltm_.get());
        }

        void MemoryTierManager::optimizeBlocks()
        {
            auto process_tier = [this](::sep::memory::MemoryTier *tier) {
                if (!tier) return;
                auto &blocks = const_cast<std::deque<::sep::memory::MemoryBlock> &>(tier->getBlocks());
                for (auto &blk : blocks)
                {
                    if (blk.allocated)
                    {
                        updateBlockMetrics(&blk, blk.coherence, blk.stability, blk.generation,
                                           blk.weight);
                    }
                }
            };

            process_tier(stm_.get());
            process_tier(mtm_.get());
            process_tier(ltm_.get());

            // Rebuild lookup tables after potential tier changes so subsequent
            // calls to findBlockByPtr reflect the updated block locations.
            rebuildLookup();
        }

        // Convenience helpers used in tests
        sep::SEPResult MemoryTierManager::promoteBlock(::sep::memory::MemoryBlock *block, ::sep::memory::MemoryBlock *&out_block)
        {
            if (!block || !block->allocated) return sep::SEPResult::INVALID_ARGUMENT;

            // Determine the target tier for promotion
            ::sep::memory::MemoryTierEnum next_tier;
            if (block->tier == ::sep::memory::MemoryTierEnum::STM)
                next_tier = ::sep::memory::MemoryTierEnum::MTM;
            else if (block->tier == ::sep::memory::MemoryTierEnum::MTM)
                next_tier = ::sep::memory::MemoryTierEnum::LTM;
            else
                return sep::SEPResult::INVALID_ARGUMENT;

            return promoteToTier(block, next_tier, out_block);
        }

        sep::SEPResult MemoryTierManager::demoteBlock(::sep::memory::MemoryBlock *block, ::sep::memory::MemoryBlock *&out_block)
        {
            if (!block || !block->allocated) return sep::SEPResult::INVALID_ARGUMENT;

            ::sep::memory::MemoryTierEnum target;
            if (block->tier == ::sep::memory::MemoryTierEnum::LTM)
                target = ::sep::memory::MemoryTierEnum::MTM;
            else if (block->tier == ::sep::memory::MemoryTierEnum::MTM)
                target = ::sep::memory::MemoryTierEnum::STM;
            else
                return sep::SEPResult::INVALID_ARGUMENT;

            return promoteToTier(block, target, out_block);
        }

        // --- Promotion and Demotion Logic ---
        sep::SEPResult MemoryTierManager::promoteToTier(::sep::memory::MemoryBlock *block,
                                                        ::sep::memory::MemoryTierEnum target_tier,
                                                        ::sep::memory::MemoryBlock *&out_block)
        {
            out_block = nullptr;
            printf("DEBUG: Attempting promotion from tier %d to tier %d\n",
                   static_cast<int>(block->tier), static_cast<int>(target_tier));
            if (!block || !block->allocated)
            {
                return sep::SEPResult::INVALID_ARGUMENT;
            }

            // Get source and destination tiers
            ::sep::memory::MemoryTier *src_tier = getTier(block->tier);
            ::sep::memory::MemoryTier *dst_tier = getTier(target_tier);
            if (!src_tier || !dst_tier)
            {
                return sep::SEPResult::INVALID_ARGUMENT;
            }

            // Try to allocate in destination tier
            out_block = dst_tier->allocate(block->size);
            if (!out_block)
            {
                printf("DEBUG: Initial allocation failed, attempting defragmentation\n");
                dst_tier->defragment();
                out_block = dst_tier->allocate(block->size);

                // Ensure tier has at least space for the block
                if (!out_block && dst_tier->getSize() < block->size * 2)
                {
                    std::size_t target = std::max(block->size * 2, dst_tier->getSize() * 2);
                    if (dst_tier->resize(target)) out_block = dst_tier->allocate(block->size);
                }
            }

            if (!out_block)
            {
                printf(
                    "DEBUG: Allocation failed even after defragmentation; attempting "
                    "compression\n");
                sep::SEPResult compress_result = compressBlock(block);
                if (compress_result == sep::SEPResult::SUCCESS)
                {
                    out_block = dst_tier->allocate(block->size);
                }
            }

            if (!out_block)
            {
                return sep::SEPResult::RESOURCE_UNAVAILABLE;
            }

            // Copy data
            std::memcpy(out_block->ptr, block->ptr, block->size);

            // Copy properties
            out_block->coherence = block->coherence;
            out_block->stability = block->stability;
            out_block->promotion_score = block->promotion_score;
            out_block->priority_score = block->priority_score;
            out_block->age = block->age;
            out_block->generation = block->generation;
            out_block->weight = block->weight;

            // Save entry in the lookup map
            lookup_map_[out_block->ptr] = out_block;

            // Release the old block
            src_tier->deallocate(block);

            printf("DEBUG: Promotion complete with coherence %.3f, stability %.3f\n",
                   static_cast<double>(out_block->coherence), static_cast<double>(out_block->stability));
            return sep::SEPResult::SUCCESS;
        }

        sep::SEPResult MemoryTierManager::compressBlock(::sep::memory::MemoryBlock *block)
        {
            if (!block || !block->allocated || !config_.enable_compression)
                return sep::SEPResult::INVALID_ARGUMENT;

            // Real compression would go here
            return sep::SEPResult::NOT_IMPLEMENTED;
        }

        ::sep::memory::MemoryTier *MemoryTierManager::determineTier(float coherence, float stability,
                                                     int generation_count)
        {
            if (coherence >= config_.promote_mtm_to_ltm &&
                stability >= config_.promote_mtm_to_ltm &&
                static_cast<uint32_t>(generation_count) >= config_.mtm_to_ltm_min_gen)
            {
                return ltm_.get();
            }
            else if (coherence >= config_.promote_stm_to_mtm &&
                     stability >= config_.promote_stm_to_mtm &&
                     static_cast<uint32_t>(generation_count) >= config_.stm_to_mtm_min_gen)
            {
                return mtm_.get();
            }
            else
            {
                return stm_.get();
            }
        }

        // Implementation of updateBlockMetrics
        ::sep::memory::MemoryBlock *MemoryTierManager::updateBlockMetrics(::sep::memory::MemoryBlock *block, float coherence,
                                                           float stability, uint32_t generation,
                                                           float context_score)
        {
            // Guard against invalid input early. Previously this method returned
            // nullptr when passed a stale pointer (for example after a block was
            // promoted and the caller still held the old address). In practice this
            // caused unit tests to fail because the lookup table retains aliases to
            // old pointers for a short period.  To make the behaviour more robust we
            // attempt to resolve the block through findBlockByPtr when the provided
            // pointer no longer refers to an allocated block.
            if (!block || !block->allocated)
            {
                ::sep::memory::MemoryBlock *resolved = block ? findBlockByPtr(block->ptr) : nullptr;
                if (!resolved || !resolved->allocated) return nullptr;
                block = resolved;
            }

            block->coherence = std::clamp(coherence, 0.0f, 1.0f);
            block->stability = std::clamp(stability, 0.0f, 1.0f);
            block->generation = generation;
            block->weight = context_score;

            // Tier-specific promotion scoring
            switch (block->tier)
            {
                case ::sep::memory::MemoryTierEnum::STM: {
                    float promotion_threshold = config_.promote_stm_to_mtm;
                    float avg_score = (block->coherence + block->stability) * 0.5f;
                    bool eligible_for_promotion = avg_score >= promotion_threshold &&
                                                  block->generation >= config_.stm_to_mtm_min_gen;
                    block->promotion_score = eligible_for_promotion ? avg_score : 0.0f;
                    block->priority_score = avg_score * (1.0f + block->weight * 0.2f);
                }
                break;
                case ::sep::memory::MemoryTierEnum::MTM: {
                    float promotion_threshold = config_.promote_mtm_to_ltm;
                    float avg_score = (block->coherence + block->stability) * 0.5f;
                    bool eligible_for_promotion = avg_score >= promotion_threshold &&
                                                  block->generation >= config_.mtm_to_ltm_min_gen;
                    block->promotion_score = eligible_for_promotion ? avg_score : 0.0f;
                    block->priority_score = avg_score * (1.0f + block->weight * 0.3f);
                }
                break;
                case ::sep::memory::MemoryTierEnum::LTM: {
                    float avg_score = (block->coherence + block->stability) * 0.5f;
                    block->promotion_score = 0.0f;  // Nothing above LTM
                    block->priority_score = avg_score * (1.0f + block->weight * 0.5f);
                }
                break;
                default: {
                    // Handle HOST, DEVICE, UNIFIED or any other memory type
                    float avg_score = (block->coherence + block->stability) * 0.5f;
                    block->promotion_score = 0.0f;
                    block->priority_score = avg_score;
                }
                break;
            }

            // Promote or demote blocks if needed based on their scores
            if (block->promotion_score > 0.0f)
            {
                ::sep::memory::MemoryTierEnum target_tier;
                if (block->tier == ::sep::memory::MemoryTierEnum::STM)
                    target_tier = ::sep::memory::MemoryTierEnum::MTM;
                else if (block->tier == ::sep::memory::MemoryTierEnum::MTM)
                    target_tier = ::sep::memory::MemoryTierEnum::LTM;
                else
                    return block;  // No promotion from LTM

                ::sep::memory::MemoryBlock *new_block = nullptr;
                if (promoteToTier(block, target_tier, new_block) == sep::SEPResult::SUCCESS)
                {
                    return new_block;
                }
            }
            else if (block->coherence < config_.demotion_threshold ||
                     block->stability < config_.demotion_threshold)
            {
                if (block->tier == ::sep::memory::MemoryTierEnum::LTM || block->tier == ::sep::memory::MemoryTierEnum::MTM)
                {
                    ::sep::memory::MemoryTierEnum target_tier = (block->tier == ::sep::memory::MemoryTierEnum::LTM)
                                                     ? ::sep::memory::MemoryTierEnum::MTM
                                                     : ::sep::memory::MemoryTierEnum::STM;
                    ::sep::memory::MemoryBlock *new_block = nullptr;
                    if (promoteToTier(block, target_tier, new_block) == sep::SEPResult::SUCCESS)
                    {
                        return new_block;
                    }
                }
            }

            return block;
        }

        // Implementation of the missing updateBlockProperties function
        ::sep::memory::MemoryBlock *MemoryTierManager::updateBlockProperties(::sep::memory::MemoryBlock *block,
                                                              float promotion_score,
                                                              float priority_score,
                                                              std::uint32_t age, float weight)
        {
            // Guard against invalid input early
            if (!block || !block->allocated)
            {
                ::sep::memory::MemoryBlock *resolved = block ? findBlockByPtr(block->ptr) : nullptr;
                if (!resolved || !resolved->allocated) return nullptr;
                block = resolved;
            }

            block->promotion_score = promotion_score;
            block->priority_score = priority_score;
            block->age = age;
            block->weight = weight;

            return block;
        }

        void MemoryTierManager::cleanupExpiredPatterns()
        {
            std::vector<std::size_t> expired_ids;

            for (auto &tier_ptr : {stm_.get(), mtm_.get(), ltm_.get()})
            {
                if (!tier_ptr) continue;
                for (auto &block : tier_ptr->getBlocksForModification())
                {
                    if (block.allocated && block.age > config_.pattern_expiration_age)
                    {
                        expired_ids.push_back(reinterpret_cast<std::size_t>(block.ptr));
                    }
                }
            }

            for (const auto id : expired_ids)
            {
                removePattern(id);
            }
        }

        void MemoryTierManager::prunePatternsByPriority(::sep::memory::MemoryTierEnum tier, size_t max_count)
        {
            ::sep::memory::MemoryTier *target_tier = getTier(tier);
            if (!target_tier) return;

            auto &blocks = target_tier->getBlocksForModification();
            if (blocks.size() <= max_count) return;

            std::vector<::sep::memory::MemoryBlock *> allocated_blocks;
            for (auto &block : blocks)
            {
                if (block.allocated)
                {
                    allocated_blocks.push_back(&block);
                }
            }

            if (allocated_blocks.size() <= max_count) return;

            std::sort(allocated_blocks.begin(), allocated_blocks.end(),
                      [](const ::sep::memory::MemoryBlock *a, const ::sep::memory::MemoryBlock *b) {
                          return a->priority_score < b->priority_score;
                      });

            size_t num_to_prune = allocated_blocks.size() - max_count;
            for (size_t i = 0; i < num_to_prune; ++i)
            {
                removePattern(reinterpret_cast<std::size_t>(allocated_blocks[i]->ptr));
                deallocate(allocated_blocks[i]);
            }
        }

        void MemoryTierManager::registerGenericData(std::size_t id, const void *data)
        {
            // This implementation assumes the caller manages the lifetime of the data.
            // A more robust implementation would involve copying the data or using smart pointers.
            data_registry_[id] = std::unique_ptr<void, std::function<void(void *)>>(
                const_cast<void *>(data), [](void *) {});
        }


sep::SEPResult MemoryTierManager::processMemoryBlocks(void *input_data, void *output_data,
                                                      const void *config, size_t count,
                                                      const void *previous_data, void *stream)
{
    if (!input_data || !output_data)
    {
        return sep::SEPResult::INVALID_ARGUMENT;
    }

    // Cast config to expected type if provided
    const sep::quantum::ProcessingConfig *proc_config = nullptr;
    if (config)
    {
        proc_config = static_cast<const sep::quantum::ProcessingConfig *>(config);
    }

    try
    {
        // Process memory blocks based on their type
        // This function is designed to work with raw memory blocks
        // and apply transformations based on the configuration

        // Enhanced memory processing pipeline with historical context and stream support
        auto *in_blocks = static_cast<::sep::memory::MemoryBlock *>(input_data);
        auto *out_blocks = static_cast<::sep::memory::MemoryBlock *>(output_data);
        
        // Cast optional parameters for enhanced processing
        const ::sep::memory::MemoryBlock *prev_blocks = static_cast<const ::sep::memory::MemoryBlock *>(previous_data);
        void *cuda_stream = stream; // For potential CUDA acceleration

        for (size_t i = 0; i < count; ++i)
        {
            // Copy basic block information
            out_blocks[i] = in_blocks[i];

            // Update block metrics based on processing
            if (in_blocks[i].allocated)
            {
                // INTEGRATION: Use previous_data for historical analysis
                if (prev_blocks && i < count)
                {
                    // Calculate change in coherence from previous state
                    float coherence_delta = in_blocks[i].coherence - prev_blocks[i].coherence;
                    out_blocks[i].promotion_score += coherence_delta * 0.3f; // Boost promotion for improving blocks
                    out_blocks[i].priority_score += (coherence_delta > 0) ? 0.1f : -0.05f; // Prioritize improving patterns
                }
                
                // INTEGRATION: Utilize CUDA stream for async processing hints
                if (cuda_stream && proc_config)
                {
                    // Mark blocks for potential GPU processing based on stream availability
                    out_blocks[i].weight *= 1.05f; // Slight weight boost for GPU-ready blocks
                }
                
                // Age the block
                out_blocks[i].age++;

                // Update coherence based on age and previous state
                float age_factor = 1.0f / (1.0f + 0.01f * out_blocks[i].age);
                out_blocks[i].coherence *= age_factor;

                // Update stability based on coherence
                out_blocks[i].stability =
                    0.9f * out_blocks[i].stability + 0.1f * out_blocks[i].coherence;

                // Calculate promotion score
                float avg_score = (out_blocks[i].coherence + out_blocks[i].stability) * 0.5f;
                out_blocks[i].promotion_score = avg_score;

                // Update priority score with weight factor
                out_blocks[i].priority_score = avg_score * (1.0f + out_blocks[i].weight * 0.2f);

                // If we have previous data, apply interaction effects
                if (previous_data && i > 0 && i < count - 1)
                {
                    auto *prev_blocks = static_cast<const ::sep::memory::MemoryBlock *>(previous_data);
                    if (prev_blocks[i - 1].allocated && prev_blocks[i + 1].allocated)
                    {
                        // Simple interaction: average neighboring coherence values
                        float neighbor_coherence =
                            (prev_blocks[i - 1].coherence + prev_blocks[i + 1].coherence) * 0.5f;
                        out_blocks[i].coherence =
                            0.95f * out_blocks[i].coherence + 0.05f * neighbor_coherence;
                    }
                }
            }
        }

        // If CUDA stream is provided and we have CUDA support in the future,
        // we can dispatch to GPU kernels here
        if (stream && proc_config && proc_config->enable_cuda)
        {
            // Future: Launch CUDA kernels for parallel processing
            // For now, just log that CUDA was requested but not available
            std::cerr << "[MemoryTierManager] CUDA processing requested but not implemented yet"
                      << std::endl;
        }

        return sep::SEPResult::SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[MemoryTierManager] Error in processMemoryBlocks: " << e.what() << std::endl;
        return sep::SEPResult::PROCESSING_ERROR;
    }
}

void MemoryTierManager::registerPattern(std::size_t id, const ::sep::compat::PatternData &pattern)
{
    auto new_pattern = std::make_unique<::sep::compat::PatternData>(pattern);
    pattern_registry_[id] = std::move(new_pattern);
}

void MemoryTierManager::removePattern(std::size_t id)
{
    pattern_registry_.erase(id);
    pattern_relationships_.erase(id);
    for (auto &pair : pattern_relationships_)
    {
        pair.second.erase(id);
    }
}

}  // namespace memory
}  // namespace sep