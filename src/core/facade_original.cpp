#include "util/memory_tier_manager.hpp"
#include "core/types.h"
#include "pattern.h"
#include "candle_data.h"
#include "result.h"
#include "core/pattern_types.h"
#include "data_parser.h"
#include "facade.h"
#include <glm/glm.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// --- Core Engine Subsystem Includes ---
// These are the real, high-performance backends.
// The Facade's job is to orchestrate them.
#include "core/pattern_metric_engine.h"
#include "core/quantum_processor.h"
#include "util/persistent_pattern_data.hpp"

namespace {

// Helper to convert core::Pattern to compat::PatternData
sep::compat::PatternData convertToCompatPattern(const sep::core::Pattern& core_pattern) {
    sep::compat::PatternData compat_pattern;
    compat_pattern.set_id(core_pattern.id);
    compat_pattern.coherence = core_pattern.coherence;
    compat_pattern.quantum_state.coherence = core_pattern.quantum_state.coherence;
    compat_pattern.quantum_state.stability = core_pattern.quantum_state.stability;
    compat_pattern.generation = core_pattern.generation;
    
    // Copy Vec4 attributes to float array
    compat_pattern[0] = core_pattern.attributes.x;
    compat_pattern[1] = core_pattern.attributes.y;
    compat_pattern[2] = core_pattern.attributes.z;
    compat_pattern[3] = core_pattern.attributes.w;
    
    return compat_pattern;
}

// Helper to convert core::CandleData to common::CandleData
sep::common::CandleData convertToCommonCandle(const sep::core::CandleData& core_candle) {
    sep::common::CandleData common_candle;
    common_candle.timestamp = core_candle.timestamp;
    common_candle.open = core_candle.open;
    common_candle.high = core_candle.high;
    common_candle.low = core_candle.low;
    common_candle.close = core_candle.close;
    common_candle.volume = core_candle.volume;
    return common_candle;
}

} // anonymous namespace

namespace sep::engine {

    // --- Private Implementation (Pimpl Idiom) ---
    // This struct holds the actual instances of our powerful subsystems,
    // completely hiding their complexity from any client of the Facade.
    struct EngineFacade::Impl
    {
        // --- Subsystem Instances ---
        std::unique_ptr<quantum::QuantumProcessor> quantum_processor;
        std::unique_ptr<quantum::PatternMetricEngine> pattern_metric_engine;
        memory::MemoryTierManager* memory_manager{nullptr};  // Singleton

        // --- State ---
        bool quantum_initialized{false};
        bool memory_initialized{false};
        uint64_t request_counter{0};

        // --- Constructor ---
        // Initializes all subsystems when the facade is created.
        Impl(const quantum::QuantumProcessor::Config& q_config,
             const memory::MemoryTierManager::Config& m_config)
        {
            quantum_processor = quantum::createQuantumProcessor(q_config);
            pattern_metric_engine = std::make_unique<quantum::PatternMetricEngine>();
            memory_manager = &memory::MemoryTierManager::getInstance();
            memory_manager->init(m_config);

            // Sub-components can also be initialized here
            pattern_metric_engine->init(nullptr);  // Init for CPU

            quantum_initialized = true;
            memory_initialized = true;
        }
    };

// --- Singleton Accessor ---
EngineFacade& EngineFacade::getInstance() {
    static EngineFacade instance;
    return instance;
}

// --- Lifecycle Management ---
core::Result EngineFacade::initialize() {
    if (initialized_) {
        return core::Result::ALREADY_EXISTS;
    }

    try
    {
        // --- Default Configurations ---
        // These can be loaded from a file or environment in a real scenario
        quantum::QuantumProcessor::Config quantum_config{};
        quantum_config.max_qubits = 10000;
        quantum_config.decoherence_rate = 0.01f;
        quantum_config.measurement_threshold = 0.65f;
        quantum_config.enable_gpu = true;  // Default to using GPU if available

        memory::MemoryTierManager::Config memory_config{};
        memory_config.stm_size = 16 << 20;   // 16MB
        memory_config.mtm_size = 64 << 20;   // 64MB
        memory_config.ltm_size = 256 << 20;  // 256MB
        memory_config.promote_stm_to_mtm = 0.7f;
        memory_config.promote_mtm_to_ltm = 0.9f;
        memory_config.demote_threshold = 0.4f;

        // Create the implementation with the configurations
        impl_ = std::make_unique<Impl>(quantum_config, memory_config);
    }
    catch (const std::exception& e)
    {
        // In a real application, you'd log this error
        return core::Result::FAILURE;
    }

    initialized_ = true;
    return core::Result::SUCCESS;
}

core::Result EngineFacade::shutdown() {
    if (!initialized_) {
        return core::Result::NOT_INITIALIZED;
    }

    if (impl_->memory_manager)
    {
        impl_->memory_manager->shutdown();
    }

    impl_.reset();
    initialized_ = false;
    return core::Result::SUCCESS;
}

// --- Core AGI/Computational Methods ---

core::Result EngineFacade::processPatterns(const PatternProcessRequest& request, 
                                          PatternProcessResponse& response) {
    if (!initialized_ || !impl_)
    {
        return core::Result::NOT_INITIALIZED;
    }

    ++impl_->request_counter;

    // This is a high-level, generic evolution step.
    // It processes a batch of already-formed patterns.
    auto batch_result = impl_->quantum_processor->processAll();

    if (!batch_result.success)
    {
        response.processing_complete = false;
        return core::Result::PROCESSING_ERROR;
    }

    // Populate the response
    for (const auto& res : batch_result.results)
    {
        response.processed_patterns.push_back(res.pattern);
    }
    response.correlation_id = "corr_" + std::to_string(impl_->request_counter);
    response.processing_complete = true;
    
    return core::Result::SUCCESS;
}

core::Result EngineFacade::analyzePattern(const PatternAnalysisRequest& request,
                                         PatternAnalysisResponse& response) {
    if (!initialized_ || !impl_)
    {
        return core::Result::NOT_INITIALIZED;
    }

    // Retrieve the target pattern
    auto target_pattern = impl_->quantum_processor->getPattern(request.pattern_id);
    if (target_pattern.id.empty())
    {
        response.confidence_score = 0.0f;
        response.analysis_summary = "Pattern not found: " + request.pattern_id;
        return core::Result::NOT_FOUND;
    }

    // Use the PatternMetricEngine to compute detailed metrics
    impl_->pattern_metric_engine->clear();
    impl_->pattern_metric_engine->addPattern(target_pattern.data);
    const auto& metrics = impl_->pattern_metric_engine->computeMetrics();

    if (metrics.empty())
    {
        response.confidence_score = 0.0f;
        response.analysis_summary = "Failed to compute metrics for pattern: " + request.pattern_id;
        return core::Result::PROCESSING_ERROR;
    }

    response.pattern = target_pattern;
    response.confidence_score = metrics[0].coherence;  // Use computed coherence as confidence
    response.analysis_summary = "Metrics computed.";
    // You could add stability and entropy to the response struct as well

    return core::Result::SUCCESS;
}

core::Result EngineFacade::processBatch(const BatchProcessRequest& request,
                                       PatternProcessResponse& response) {
    if (!initialized_ || !impl_)
    {
        return core::Result::NOT_INITIALIZED;
    }

    ++impl_->request_counter;

    // Step 1: Use the DataParser to convert raw data (Candles) into engine Patterns
    DataParser parser;
    // Convert core::CandleData to common::CandleData
    std::vector<sep::common::CandleData> common_candles;
    common_candles.reserve(request.market_data.size());
    
    for (const auto& candle : request.market_data) {
        common_candles.push_back(convertToCommonCandle(candle));
    }
    
    auto patterns = parser.candlesToPatterns(common_candles);

    if (patterns.empty())
    {
        response.processing_complete = false;
        return core::Result::INVALID_ARGUMENT;
    }

    // Step 2: Add all new patterns to the quantum processor's state
    for (const auto& p : patterns)
    {
        impl_->quantum_processor->addPattern(p);
    }

    // Step 3: Evolve the entire system one step with the new data
    auto batch_result = impl_->quantum_processor->processAll();
    if (!batch_result.success)
    {
        response.processing_complete = false;
        return core::Result::PROCESSING_ERROR;
    }

    response.processed_patterns.clear();
    response.processed_patterns.reserve(patterns.size());
    
    for (const auto& p : batch_result.results) {
        response.processed_patterns.push_back(p.pattern);
    }
    response.processed_patterns = impl_->quantum_processor->getPatterns();
    response.correlation_id = "batch_" + std::to_string(impl_->request_counter);
    response.processing_complete = true;
    
    return core::Result::SUCCESS;
}

core::Result EngineFacade::storePattern(const StorePatternRequest& request,
                                      StorePatternResponse& response) {
    if (!initialized_ || !impl_ || !impl_->memory_manager) {
        return core::Result::NOT_INITIALIZED;
    }

    try {
        // Determine appropriate memory tier based on pattern metrics
        auto target_tier = impl_->memory_manager->determineTier(
            request.coherence,
            request.stability,
            request.generation_count
        );

        if (!target_tier) {
            response.success = false;
            response.error_message = "Failed to determine appropriate memory tier";
            return core::Result::INVALID_ARGUMENT;
        }

        // Allocate memory block in the determined tier
        auto block = impl_->memory_manager->allocate(
            sizeof(core::Pattern),  // Size of pattern data
            target_tier->getTierEnum()
        );

        if (!block) {
            response.success = false;
            response.error_message = "Failed to allocate memory block";
            return core::Result::OUT_OF_MEMORY;
        }

        // Store pattern data and register it
        impl_->memory_manager->registerPattern(
            std::hash<std::string>{}(request.pattern.id),
            convertToCompatPattern(request.pattern)
        );

        // Update block metrics with pattern properties
        block = impl_->memory_manager->updateBlockMetrics(
            block,
            request.coherence,
            request.stability,
            request.generation_count,
            request.weight  // Use weight as context score
        );

        if (!block) {
            response.success = false;
            response.error_message = "Failed to update block metrics";
            return core::Result::PROCESSING_ERROR;
        }

        response.success = true;
        return core::Result::SUCCESS;

    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = std::string("Exception during pattern storage: ") + e.what();
        return core::Result::FAILURE;
    }
}

// --- Memory and Health Methods ---

core::Result EngineFacade::queryMemory(const MemoryQueryRequest& request,
                                      std::vector<core::Pattern>& results) {
    if (!initialized_ || !impl_ || !impl_->memory_manager)
    {
        return core::Result::NOT_INITIALIZED;
    }

    // This is a simplified query. A real one would parse request.query_id
    // to build complex criteria (e.g., "coherence > 0.8 AND stability < 0.5")
    results.clear();

    auto stm_patterns = impl_->quantum_processor->getPatternsByTier(sep::memory::MemoryTierEnum::STM);
    auto mtm_patterns = impl_->quantum_processor->getPatternsByTier(sep::memory::MemoryTierEnum::MTM);
    auto ltm_patterns = impl_->quantum_processor->getPatternsByTier(sep::memory::MemoryTierEnum::LTM);

    results.insert(results.end(), stm_patterns.begin(), stm_patterns.end());
    results.insert(results.end(), mtm_patterns.begin(), mtm_patterns.end());
    results.insert(results.end(), ltm_patterns.begin(), ltm_patterns.end());

    return core::Result::SUCCESS;
}

core::Result EngineFacade::getHealthStatus(HealthStatusResponse& response) {
    response.engine_healthy = initialized_;
    if (impl_)
    {
        response.quantum_systems_ready = impl_->quantum_initialized;
        response.memory_systems_ready = impl_->memory_initialized;
        response.status_message = initialized_ ? "Engine operational" : "Engine not initialized";
    }
    else
    {
        response.status_message = "Engine not initialized";
    }
    // Placeholders for real system monitoring
    response.cpu_usage = 25.0f;
    response.memory_usage = 512.0f;  // in MB

    return core::Result::SUCCESS;
}

core::Result EngineFacade::getMemoryMetrics(MemoryMetricsResponse& response) {
    if (!initialized_ || !impl_ || !impl_->memory_manager)
    {
        return core::Result::NOT_INITIALIZED;
    }

    response.stm_utilization =
        impl_->memory_manager->getTierUtilization(sep::memory::MemoryTierEnum::STM);
    response.mtm_utilization =
        impl_->memory_manager->getTierUtilization(sep::memory::MemoryTierEnum::MTM);
    response.ltm_utilization =
        impl_->memory_manager->getTierUtilization(sep::memory::MemoryTierEnum::LTM);
    response.total_patterns = impl_->quantum_processor->getPatternCount();
    // active_patterns could be defined as patterns in MTM or LTM
    response.active_patterns =
        impl_->quantum_processor->getPatternsByTier(sep::memory::MemoryTierEnum::MTM).size() +
        impl_->quantum_processor->getPatternsByTier(sep::memory::MemoryTierEnum::LTM).size();

    return core::Result::SUCCESS;
}

core::Result EngineFacade::qfhAnalyze(const QFHAnalysisRequest& request,
                                     QFHAnalysisResponse& response) {
    if (!initialized_ || !impl_)
    {
        return core::Result::NOT_INITIALIZED;
    }

    try {
        // Use the PatternMetricEngine to analyze the bitstream
        impl_->pattern_metric_engine->clear();
        
        // Convert bitstream to pattern data format for analysis
        // This is a simplified conversion - in practice you'd have more sophisticated mapping
        compat::PatternData pattern_data;
        pattern_data.set_id("qfh_bitstream_" + std::to_string(impl_->request_counter++));
        
        // Map bitstream characteristics to pattern attributes
        float ones_ratio = 0.0f;
        for (uint8_t bit : request.bitstream) {
            if (bit == 1) ones_ratio += 1.0f;
        }
        ones_ratio /= request.bitstream.size();
        
        pattern_data.coherence = ones_ratio; // Simple coherence estimate
        pattern_data.quantum_state.coherence = ones_ratio;
        pattern_data.quantum_state.stability = 1.0f - std::abs(ones_ratio - 0.5f) * 2.0f;
        
        impl_->pattern_metric_engine->addPattern(pattern_data);
        const auto& metrics = impl_->pattern_metric_engine->computeMetrics();
        
        if (!metrics.empty()) {
            response.rupture_ratio = metrics[0].entropy;  // Use entropy as rupture measure
            response.flip_ratio = ones_ratio;
            response.collapse_detected = (metrics[0].stability < 0.3f);
        } else {
            response.rupture_ratio = 0.0f;
            response.flip_ratio = ones_ratio;
            response.collapse_detected = false;
        }
        
        return core::Result::SUCCESS;
        
    } catch (const std::exception& e) {
        response.rupture_ratio = 0.0f;
        response.flip_ratio = 0.0f;
        response.collapse_detected = false;
        return core::Result::PROCESSING_ERROR;
    }
}

core::Result EngineFacade::manifoldOptimize(const ManifoldOptimizationRequest& request,
                                          ManifoldOptimizationResponse& response) {
    if (!initialized_ || !impl_)
    {
        return core::Result::NOT_INITIALIZED;
    }

    try {
        // Retrieve the target pattern
        auto target_pattern = impl_->quantum_processor->getPattern(request.pattern_id);
        if (target_pattern.id.empty())
        {
            response.success = false;
            response.pattern_id = request.pattern_id;
            return core::Result::NOT_FOUND;
        }

        // Use quantum processor to optimize the pattern toward target metrics
        auto optimization_result = impl_->quantum_processor->optimizePattern(
            target_pattern,
            request.target_coherence,
            request.target_stability
        );

        if (optimization_result.success) {
            response.success = true;
            response.pattern_id = request.pattern_id;
            response.optimized_coherence = optimization_result.final_coherence;
            response.optimized_stability = optimization_result.final_stability;
        } else {
            response.success = false;
            response.pattern_id = request.pattern_id;
            response.optimized_coherence = target_pattern.coherence;
            response.optimized_stability = target_pattern.quantum_state.stability;
        }
        
        return core::Result::SUCCESS;
        
    } catch (const std::exception& e) {
        response.success = false;
        response.pattern_id = request.pattern_id;
        response.optimized_coherence = 0.0f;
        response.optimized_stability = 0.0f;
        return core::Result::PROCESSING_ERROR;
    }
}

}  // namespace sep::engine