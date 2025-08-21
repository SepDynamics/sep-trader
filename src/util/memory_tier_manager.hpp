#pragma once

#include <array>

/**

 * @brief Manager for STM/MTM/LTM memory tiers.

 *

 * Provides global access to tiered memory pools, handling allocation,

 * promotion and demotion of MemoryBlock instances.

 */



// JSON library needs to be included before project headers

#include "util/nlohmann_json_safe.h"



// Project includes

#include "core/common.h"

#include "core/dag_graph.h"

#include "core/pattern_types.h"

#include "core/standard_includes.h"

#include "core/system_hooks.h"

#include "core/types.h"

#include "util/memory_tier.hpp"

#include "util/memory_tier_manager_serialization.hpp"

#include "util/persistent_pattern_data.hpp"

#ifndef SEP_NO_REDIS

#include "util/redis_manager.h"

#endif // SEP_NO_REDIS



// Standard library includes

#include <cstddef>

#include <cstdint>

#include <glm/vec3.hpp>

#include <memory>

#include <mutex>

#include <system_error>

#include <unordered_map>

#include <vector>





namespace sep {



namespace memory {



    using MemoryTierEnum = ::sep::memory::MemoryTierEnum;

    using PersistentPatternData = ::sep::persistence::PersistentPatternData;



    class MemoryTierManager

    {

    public:

        using Config = MemoryThresholdConfig;



        // Singleton access

        static MemoryTierManager &getInstance();



        MemoryTierManager();

        explicit MemoryTierManager(const Config &cfg);

        ~MemoryTierManager();



        void init(const Config &config);

        void shutdown();



        // Memory block allocation and management

        ::sep::memory::MemoryBlock *allocate(std::size_t size, MemoryTierEnum tier);

        void deallocate(::sep::memory::MemoryBlock *block);

        ::sep::memory::MemoryBlock *findBlockByPtr(void *ptr);



        // Tier management

        ::sep::memory::MemoryTier *getTier(MemoryTierEnum tier);

        float getTierUtilization(MemoryTierEnum tier) const;

        float getTierFragmentation(MemoryTierEnum tier) const;

        float getTotalUtilization() const;

        float getTotalFragmentation() const;

        void defragmentTier(MemoryTierEnum tier);

        void optimizeBlocks();

        void optimizeTiers();



        // Aggregate metrics

        std::size_t getTotalAllocated() const;



        // Access tier objects

        ::sep::memory::MemoryTier &getSTM();

        ::sep::memory::MemoryTier &getMTM();

        ::sep::memory::MemoryTier &getLTM();



        // Block promotion/demotion

        sep::SEPResult promoteBlock(::sep::memory::MemoryBlock *block, ::sep::memory::MemoryBlock *&out_block);

        sep::SEPResult demoteBlock(::sep::memory::MemoryBlock *block, ::sep::memory::MemoryBlock *&out_block);

        ::sep::memory::MemoryTier *determineTier(float coherence, float stability, int generation_count);

        ::sep::memory::MemoryBlock *updateBlockProperties(::sep::memory::MemoryBlock *block, float promotion_score,

                                           float priority_score, std::uint32_t age = 0,

                                           float weight = 0.0f);

        ::sep::memory::MemoryBlock *updateBlockMetrics(::sep::memory::MemoryBlock *block, float coherence, float stability,

                                        uint32_t generation, float context_score);

        void rebuildLookup();



        // Pattern management



        sep::SEPResult processMemoryBlocks(void *input_data, void *output_data, const void *config,

                                           size_t count, const void *previous_data, void *stream);



        // Generic relationship management functions

        void updateGenericRelationship(std::size_t id_a, std::size_t id_b, float strength);

        void removeDataEntry(std::size_t id);

        void pruneWeakRelationships();

        void calculateRelationshipScores();

        void loadDataFromPersistence();

        void storeDataToPersistence(const void *data,

                                    const sep::persistence::PersistentPatternData &metadata);

        void *findDataById(std::size_t id);

        const void *findDataById(std::size_t id) const;

        void registerGenericData(std::size_t id, const void *data);

        const void *getRegisteredData(std::size_t id) const;

        void cleanupExpiredData();

        void pruneDataByPriority(MemoryTierEnum tier, size_t max_count);



        // Pattern management

        void registerPattern(std::size_t id, const ::sep::compat::PatternData &pattern);

        const ::sep::compat::PatternData *getPatternData(std::size_t id) const;

        void removePattern(std::size_t id);

        void updateRelationship(std::size_t id_a, std::size_t id_b, float strength);

        void cleanupExpiredPatterns();

        void prunePatternsByPriority(MemoryTierEnum tier, size_t max_count);

        void calculateRelationshipCoherence();



        // Test helpers

        void resetForTesting(const MemoryTier::Config &cfg = MemoryTier::Config());



        dag::DagGraph &getDagGraph() { return dag_graph_; }



    private:

        static std::unique_ptr<MemoryTierManager> instance_;

        static std::once_flag once_flag_;



        MemoryTier::Config config_;

        std::unique_ptr<::sep::memory::MemoryTier> stm_;

        std::unique_ptr<::sep::memory::MemoryTier> mtm_;

        std::unique_ptr<::sep::memory::MemoryTier> ltm_;

        std::unordered_map<void *, ::sep::memory::MemoryBlock *> lookup_map_;

        // Legacy pointer lookup table used during tier transitions. When blocks

        // move between tiers the old pointer remains valid for a short period so

        // tests can resolve both the new and previous addresses. This map stores

        // those temporary associations until the next rebuild. The entries are

        // cleared whenever rebuildLookup() is invoked to keep stale pointers from

        // accumulating across multiple promotions or defragmentation cycles.

        std::unordered_map<void *, ::sep::memory::MemoryBlock *> legacy_lookup_map_;



    private:

        dag::DagGraph dag_graph_;

        std::unordered_map<std::size_t, uint64_t> pattern_dag_map_;

        core::SystemHooks *hooks_{nullptr};



        // Generic data registry using void* to avoid quantum/pattern dependencies

        std::unordered_map<std::size_t, std::unique_ptr<void, std::function<void(void *)>>>

            data_registry_;

        std::unordered_map<std::size_t, std::unordered_map<std::size_t, float>> data_relationships_;



        // Pattern specific registries

        std::unordered_map<std::size_t, std::unique_ptr<::sep::compat::PatternData>>

            pattern_registry_;

        std::unordered_map<std::size_t, std::unordered_map<std::size_t, float>>

            pattern_relationships_;



        sep::SEPResult promoteToTier(::sep::memory::MemoryBlock *block, MemoryTierEnum tier,

                                     ::sep::memory::MemoryBlock *&out_block);

        sep::SEPResult compressBlock(::sep::memory::MemoryBlock *block);



        // Generic data scoring methods for tier transition

        bool checkScoreForPromotion(float score, ::sep::memory::MemoryTier *target_tier) const;

        bool checkScoreForDemotion(float score) const;

    };



} // namespace memory

} // namespace sep



namespace nlohmann {

template <>

struct adl_serializer<sep::memory::MemoryThresholdConfig> {

    static void to_json(json& j, const sep::memory::MemoryThresholdConfig& c) {

        j = json{{"promote_stm_to_mtm", c.promote_stm_to_mtm},

                 {"promote_mtm_to_ltm", c.promote_mtm_to_ltm},

                 {"demote_threshold", c.demote_threshold},

                 {"fragmentation_threshold", c.fragmentation_threshold},

                 {"stm_size", c.stm_size},

                 {"mtm_size", c.mtm_size},

                 {"ltm_size", c.ltm_size},

                 {"stm_to_mtm_min_gen", c.stm_to_mtm_min_gen},

                 {"mtm_to_ltm_min_gen", c.mtm_to_ltm_min_gen},

                 {"use_unified_memory", c.use_unified_memory},

                 {"enable_compression", c.enable_compression},

                 {"pattern_expiration_age", c.pattern_expiration_age}};

   }



    static void from_json(const json& j, sep::memory::MemoryThresholdConfig& c) {

        j.at("promote_stm_to_mtm").get_to(c.promote_stm_to_mtm);

        j.at("promote_mtm_to_ltm").get_to(c.promote_mtm_to_ltm);

        j.at("demote_threshold").get_to(c.demote_threshold);

        j.at("fragmentation_threshold").get_to(c.fragmentation_threshold);

        j.at("stm_size").get_to(c.stm_size);

        j.at("mtm_size").get_to(c.mtm_size);

        j.at("ltm_size").get_to(c.ltm_size);

        j.at("stm_to_mtm_min_gen").get_to(c.stm_to_mtm_min_gen);

        j.at("mtm_to_ltm_min_gen").get_to(c.mtm_to_ltm_min_gen);

        j.at("use_unified_memory").get_to(c.use_unified_memory);

        j.at("enable_compression").get_to(c.enable_compression);

        if (j.contains("pattern_expiration_age")) {

            j.at("pattern_expiration_age").get_to(c.pattern_expiration_age);

        }

    }

};

} // namespace nlohmann


