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
#include "core/quantum_types.h"

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

/**
 * Quantum threshold configuration for quantum processing systems
 */
struct QuantumThresholdConfig {
    double coherence_threshold = 0.7;
    double stability_threshold = 0.8;
    double collapse_threshold = 0.3;
    double entanglement_threshold = 0.6;
    double phase_threshold = 0.9;
    double ltm_coherence_threshold = 0.9;
    double mtm_coherence_threshold = 0.75;
    int max_iterations = 1000;
    bool enable_adaptive_thresholds = true;
    double adaptive_learning_rate = 0.01;
};

// Originally from src/core/types.h
struct PinState {
    uint64_t pin_id{0};
    double value{0.0};
    float coherence{0.0f};
    uint64_t tick{0};
    std::vector<uint32_t> bits{};
};

// quantum namespace types moved to quantum_types.h to avoid duplicates

namespace compat {
    // Alias for backward compatibility or for a stable API
    using Pattern = quantum::Pattern;
} // namespace compat

namespace config {
    struct SystemConfig {
        bool debug_mode = false;
        uint32_t max_threads = 8;
        uint32_t memory_pool_size_mb = 1024;
        std::string log_level = "INFO";
        std::string data_path = "./data";
        
        // Additional fields expected by manager.cpp
        struct {
            size_t pool_size_mb = 1024;
            bool enable_caching = true;
        } memory;
        
        struct {
            double coherence_threshold = 0.7;
            bool enable_quantum_processing = true;
        } quantum;
    };
    
    struct CudaConfig {
        bool use_gpu = true;
        int device_id = 0;
        size_t memory_limit_mb = 4096;
        int max_memory_mb = 8192;
        int batch_size = 1024;
        float gpu_memory_limit = 0.9f;
        bool enable_peer_access = false;
        bool enable_unified_memory = true;
        uint32_t stream_pool_size = 4;
        bool enable_profiling = false;
    };
}

} // namespace sep

#endif // SRC_CORE_TYPES_MERGED_H
