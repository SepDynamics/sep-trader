#pragma once

#include "util/compiler.h"
#include <vector>

namespace dsl::stdlib {

// Import Value and Context types
using dsl::compiler::Value;
using dsl::compiler::Context;

// Initialize the DSL engine components
void initialize_engine_components();

// Core primitive helpers
Value generate_sine_wave(const std::vector<Value>& args);
Value weighted_sum(const std::vector<Value>& args);

// Register available primitives with the runtime context
void register_core_primitives(Context& context);

} // namespace dsl::stdlib

