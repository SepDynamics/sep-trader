#pragma once
#include "compiler/compiler.h"
#include <vector>
#include <memory>

namespace dsl::stdlib {

// Import Value and Context types
using dsl::compiler::Value;
using dsl::compiler::Context;

// Forward declarations for SEP engine integration
namespace engine {
    struct Pattern;
    struct QFHResult;
    struct QBSAResult;
}

/**
 * Phase 2: Core Computational Primitives from TASK.md
 * These wrap the existing SEP engine functionality with DSL-friendly interfaces
 */

// ============================================================================
// Type Checking & Conversion Functions (TASK.md Phase 2A Priority 1)
// ============================================================================

/// Check if value is a number
Value is_number(const std::vector<Value>& args);

/// Check if value is a string
Value is_string(const std::vector<Value>& args);

/// Check if value is a boolean
Value is_bool(const std::vector<Value>& args);

/// Convert any value to string
Value to_string(const std::vector<Value>& args);

/// Convert to number with error handling
Value to_number(const std::vector<Value>& args);

// ============================================================================
// Pattern Operations
// ============================================================================

/// Create pattern from data stream
Value create_pattern(const std::vector<Value>& args);

/// Evolve pattern through time
Value evolve_pattern(const std::vector<Value>& args);

/// Merge two patterns into one
Value merge_patterns(const std::vector<Value>& args);

/// Measure coherence of a pattern
Value measure_coherence(const std::vector<Value>& args);

/// Measure stability of a pattern  
Value measure_stability(const std::vector<Value>& args);

/// Measure entropy of a pattern
Value measure_entropy(const std::vector<Value>& args);

// ============================================================================
// Quantum Operations  
// ============================================================================

/// QFH analysis on bitstream
Value qfh_analyze(const std::vector<Value>& args);

/// QBSA analysis on pattern
Value qbsa_analyze(const std::vector<Value>& args);

/// Manifold optimization with constraints
Value manifold_optimize(const std::vector<Value>& args);

/// Detect pattern collapse
Value detect_collapse(const std::vector<Value>& args);

// ============================================================================
// Memory Operations
// ============================================================================

/// Store pattern in memory tier
Value store_pattern(const std::vector<Value>& args);

/// Retrieve pattern by ID
Value retrieve_pattern(const std::vector<Value>& args);

/// Promote pattern to higher tier
Value promote_pattern(const std::vector<Value>& args);

/// Query patterns by criteria
Value query_patterns(const std::vector<Value>& args);

// ============================================================================
// Stream Operations (Extension from TASK.md)
// ============================================================================

/// Extract bits from data stream
Value extract_bits(const std::vector<Value>& args);

/// Create weighted sum with coefficients
Value weighted_sum(const std::vector<Value>& args);

/// Generate sine wave for testing
Value generate_sine_wave(const std::vector<Value>& args);

// ============================================================================
// Registration Function
// ============================================================================

/// Register all core primitives with the runtime context
void register_core_primitives(Context& context);

} // namespace dsl::stdlib
