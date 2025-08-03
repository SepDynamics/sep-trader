#pragma once
#include "../core_primitives.h"
#include <vector>

namespace dsl::stdlib {

// Import Value type
using dsl::compiler::Value;
using dsl::compiler::Context;

// ============================================================================
// Basic Statistical Functions (TASK.md Phase 2A Priority 1)
// ============================================================================

/// Calculate mean (average) of data array
Value mean_func(const std::vector<Value>& args);

/// Calculate median (middle value) of data array
Value median_func(const std::vector<Value>& args);

/// Calculate standard deviation of data array
Value std_dev_func(const std::vector<Value>& args);

/// Calculate variance of data array
Value variance_func(const std::vector<Value>& args);

/// Find minimum value in data array
Value min_value_func(const std::vector<Value>& args);

/// Find maximum value in data array
Value max_value_func(const std::vector<Value>& args);

// ============================================================================
// Advanced Statistical Functions
// ============================================================================

/// Calculate Pearson correlation between two data arrays
Value correlation_func(const std::vector<Value>& args);

/// Calculate covariance between two data arrays
Value covariance_func(const std::vector<Value>& args);

/// Calculate nth percentile of data array
Value percentile_func(const std::vector<Value>& args);

// ============================================================================
// Registration Function
// ============================================================================

/// Register all statistical functions with the runtime context
void register_statistical(Context& context);

}
