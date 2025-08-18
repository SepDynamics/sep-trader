#pragma once

#ifndef SRC_CORE_TYPES_MERGED_H
#define SRC_CORE_TYPES_MERGED_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>
#include <complex>
#include <glm/glm.hpp>

#include "core/pattern_types.h"
#include "core/standard_includes.h"
#include "core/config.h"

namespace sep {

// Originally from src/util/types.h
namespace memory {

    // Memory tier types that are used across multiple modules
    enum class MemoryTierEnum : int
    {
        // Logical memory tiers
        STM,
        SHORT_TERM = STM,
        SHORT_TERM_MEMORY = SHORT_TERM,
        MTM,
        LTM,
        // Physical memory locations
        HOST = 100,    // Host memory (CPU)
        DEVICE = 101,  // Device memory (GPU)
        UNIFIED = 102  // Unified memory (accessible by both CPU and GPU)
    };

    // Convert string to MemoryTierEnum
    inline MemoryTierEnum stringToMemoryTier(const std::string& tier)
    {
        if (tier == "stm") return MemoryTierEnum::STM;
        if (tier == "mtm") return MemoryTierEnum::MTM;
        if (tier == "ltm") return MemoryTierEnum::LTM;
        if (tier == "host") return MemoryTierEnum::HOST;
        if (tier == "device") return MemoryTierEnum::DEVICE;
        if (tier == "unified") return MemoryTierEnum::UNIFIED;
        throw std::invalid_argument(std::string("Invalid memory tier string: ") + tier.c_str());
    }

    enum class CompressionMethod : std::uint8_t {
        None,
        ZSTD,
        // Add other methods as needed
    };

    struct MemoryThresholdConfig {
        std::size_t stm_size;
        std::size_t mtm_size;
        std::size_t ltm_size;
        float promote_stm_to_mtm;
        float promote_mtm_to_ltm;
        float demote_threshold;
        float fragmentation_threshold;
        bool use_unified_memory;
        bool enable_compression;
        int stm_to_mtm_min_gen;
        int mtm_to_ltm_min_gen;
        std::uint32_t pattern_expiration_age{1000};
    };

} // namespace memory

// Originally from src/util/types.h
namespace common {

    struct TraceConfig {
        bool enable_trace_logging;
        std::string trace_log_file;
        int trace_level;
    };

} // namespace common

// Originally from src/core/internal/types.h
namespace config {
    struct SystemConfig;
    struct CudaConfig;
    struct LogConfig;
    struct AnalyticsConfig;
}

// Originally from src/core/types.h
struct PinState {
    uint64_t pin_id{0};
    double value{0.0};
    float coherence{0.0f};
    uint64_t tick{0};
    std::vector<uint32_t> bits{};
};

namespace quantum {

    struct QuantumState {
        float coherence{0.0f};
        float stability{0.0f};
        float entropy{0.0f};
        float mutation_rate{0.0f};
        float evolution_rate{0.0f};
        float energy{0.0f};
        float coupling_strength{0.0f};
        int generation{0};
        int mutation_count{0};
        sep::memory::MemoryTierEnum memory_tier{sep::memory::MemoryTierEnum::STM};
        int access_frequency{0};

        enum class Status {
            STABLE,
            UNSTABLE,
            COLLAPSED
        };
        Status state{Status::STABLE};
        float phase{0.0f};
    };

    enum class RelationshipType {
        Generic,
        Entanglement,
        ENTANGLEMENT = Entanglement,  // Add alias for compatibility
        Causality
    };

    struct PatternRelationship {
        std::string targetId{};
        float strength{0.0f};
        RelationshipType type{RelationshipType::Generic};
    };

    struct Pattern {
        std::string id;
        glm::vec4 position{0.0f};
        glm::vec3 momentum{0.0f};
        QuantumState quantum_state{};
        std::vector<PatternRelationship> relationships{};
        sep::compat::PatternData data{};
        std::vector<std::string> parent_ids{};
        uint64_t timestamp{0};
        uint64_t last_accessed{0};
        uint64_t last_modified{0};
        int generation{0};
        float coherence{0.0f};
        glm::vec4 velocity{0.0f};
        glm::vec4 attributes{0.0f};
        std::complex<float> amplitude{1.0f, 0.0f};
        QuantumState state{};
        uint64_t last_updated{0};
    };

} // namespace quantum

namespace compat {
    // Alias for backward compatibility or for a stable API
    using Pattern = quantum::Pattern;
} // namespace compat

} // namespace sep

#endif // SRC_CORE_TYPES_MERGED_H
