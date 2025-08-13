#pragma once
#include <vector>

#include "core_primitives.h"
#include "vm.h"

namespace sep_dsl::stdlib {

// Import Value type
using sep_dsl::bytecode::Value;
using Context = sep_dsl::bytecode::VMExecutionContext;

// ============================================================================
// Basic Arithmetic Functions (TASK.md Phase 2A Priority 1)
// ============================================================================

/// Absolute value
Value abs_func(const std::vector<Value>& args);

/// Minimum of two values
Value min_func(const std::vector<Value>& args);

/// Maximum of two values
Value max_func(const std::vector<Value>& args);

/// Round to nearest integer
Value round_func(const std::vector<Value>& args);

/// Floor function
Value floor_func(const std::vector<Value>& args);

/// Ceiling function
Value ceil_func(const std::vector<Value>& args);

// ============================================================================
// Trigonometric Functions
// ============================================================================

/// Sine function
Value sin_func(const std::vector<Value>& args);

/// Cosine function
Value cos_func(const std::vector<Value>& args);

/// Tangent function
Value tan_func(const std::vector<Value>& args);

/// Arcsine function
Value asin_func(const std::vector<Value>& args);

/// Arccosine function
Value acos_func(const std::vector<Value>& args);

/// Arctangent function
Value atan_func(const std::vector<Value>& args);

// ============================================================================
// Exponential/Logarithmic Functions
// ============================================================================

/// Exponential (e^x)
Value exp_func(const std::vector<Value>& args);

/// Natural logarithm
Value log_func(const std::vector<Value>& args);

/// Base-10 logarithm
Value log10_func(const std::vector<Value>& args);

/// Power function (x^y)
Value pow_func(const std::vector<Value>& args);

/// Square root
Value sqrt_func(const std::vector<Value>& args);

// ============================================================================
// Registration Function
// ============================================================================

/// Register all math functions with the runtime context
void register_math(Context& context);

}
