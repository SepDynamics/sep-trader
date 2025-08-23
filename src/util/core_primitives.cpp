#include "util/core_primitives.h"
#include "util/pattern_processing.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdint>

#include <unordered_map>
#include <algorithm>
#include <bitset>
#include <sstream>
#include <random>

namespace dsl::stdlib {

// Real SEP engine components for DSL integration
// Component management moved to EngineFacade to eliminate global state
// Static storage for patterns and simple metadata
static std::unordered_map<std::string, Value> pattern_store;
static std::unordered_map<std::string, std::string> pattern_tiers;
static std::mt19937 rng(42); // Deterministic for testing
static int next_pattern_id = 1;

void initialize_engine_components() {
    // Component initialization moved to EngineFacade to eliminate global state
    // This function is now a no-op and can be removed in future refactoring
    std::cout << "initialize_engine_components: Component management handled by EngineFacade" << std::endl;
}

// ============================================================================
// Pattern Operations - Now handled via EngineFacade
// ============================================================================

// ============================================================================
// Quantum Operations - Now handled via EngineFacade
// ============================================================================

// ============================================================================
// Memory Operations - Now handled via EngineFacade
// ============================================================================

// ============================================================================
// Type Checking & Conversion Functions - Now handled via interpreter builtin functions
// ============================================================================

// ============================================================================
// Math Functions - Now handled via interpreter builtin functions
// ============================================================================

// ============================================================================
// Statistical Functions - Now handled via interpreter builtin functions
// ============================================================================

Value generate_sine_wave(const std::vector<Value>& args) {
    double frequency = 10.0; // Default 10Hz
    if (!args.empty() && std::holds_alternative<double>(args[0])) {
        frequency = std::get<double>(args[0]);
    }

    std::ostringstream oss;
    for (int i = 0; i < 100; ++i) {
        double val = std::sin(2 * M_PI * frequency * i / 100.0);
        if (i) oss << ',';
        oss << val;
    }
    return Value(oss.str());
}

Value weighted_sum(const std::vector<Value>& args) {
    if (args.size() % 2 != 0 || args.empty()) {
        throw std::runtime_error("weighted_sum requires value/weight pairs");
    }
    double total = 0.0;
    double weights = 0.0;
    for (size_t i = 0; i < args.size(); i += 2) {
        if (!std::holds_alternative<double>(args[i]) || !std::holds_alternative<double>(args[i + 1])) {
            throw std::runtime_error("weighted_sum arguments must be numbers");
        }
        double v = std::get<double>(args[i]);
        double w = std::get<double>(args[i + 1]);
        total += v * w;
        weights += w;
    }
    return Value(weights == 0.0 ? 0.0 : total / weights);
}

// ============================================================================
// Registration Function
// ============================================================================

void register_core_primitives(Context& context) {
    // Only register functions that are not duplicated in the interpreter
    context.set_function("generate_sine_wave", generate_sine_wave);
    context.set_function("weighted_sum", weighted_sum);
    
    std::cout << "Registered " << 2 << " core primitive functions (including trading functions)" << std::endl;
}

} // namespace dsl::stdlib
