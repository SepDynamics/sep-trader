#include "core_primitives.h"
#include <iostream>
#include <cmath>
#include <random>
#include <unordered_map>

namespace dsl::stdlib {

// Static storage for patterns (in real implementation, this would interface with SEP engine)
static std::unordered_map<std::string, Value> pattern_store;
static std::mt19937 rng(42); // Deterministic for testing

// ============================================================================
// Pattern Operations
// ============================================================================

Value create_pattern(const std::vector<Value>& args) {
    std::cout << "Creating pattern from data stream..." << std::endl;
    
    // Mock pattern creation - in real implementation, call into SEP engine
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double coherence = dist(rng);
    double stability = dist(rng); 
    double entropy = dist(rng);
    
    // Return pattern ID for now
    std::string pattern_id = "pattern_" + std::to_string(pattern_store.size());
    pattern_store[pattern_id + "_coherence"] = Value(coherence);
    pattern_store[pattern_id + "_stability"] = Value(stability);
    pattern_store[pattern_id + "_entropy"] = Value(entropy);
    
    return Value(pattern_id);
}

Value evolve_pattern(const std::vector<Value>& args) {
    if (args.size() < 2) {
        throw std::runtime_error("evolve_pattern requires pattern and time arguments");
    }
    
    std::cout << "Evolving pattern through time..." << std::endl;
    
    // Mock evolution - in real implementation, call QuantumManifoldOptimizationEngine
    return args[0]; // Return evolved pattern
}

Value merge_patterns(const std::vector<Value>& args) {
    if (args.size() < 2) {
        throw std::runtime_error("merge_patterns requires two pattern arguments");
    }
    
    std::cout << "Merging patterns..." << std::endl;
    
    // Mock merge - in real implementation, call pattern merger
    return create_pattern({}); // Return new merged pattern
}

Value measure_coherence(const std::vector<Value>& args) {
    std::cout << "Measuring pattern coherence..." << std::endl;
    
    if (!args.empty() && args[0].type == Value::STRING) {
        std::string pattern_id = args[0].get<std::string>();
        auto it = pattern_store.find(pattern_id + "_coherence");
        if (it != pattern_store.end()) {
            return it->second;
        }
    }
    
    // Default mock coherence
    std::uniform_real_distribution<double> dist(0.6, 0.95);
    return Value(dist(rng));
}

Value measure_stability(const std::vector<Value>& args) {
    std::cout << "Measuring pattern stability..." << std::endl;
    
    if (!args.empty() && args[0].type == Value::STRING) {
        std::string pattern_id = args[0].get<std::string>();
        auto it = pattern_store.find(pattern_id + "_stability");
        if (it != pattern_store.end()) {
            return it->second;
        }
    }
    
    // Default mock stability  
    std::uniform_real_distribution<double> dist(0.5, 0.9);
    return Value(dist(rng));
}

Value measure_entropy(const std::vector<Value>& args) {
    std::cout << "Measuring pattern entropy..." << std::endl;
    
    if (!args.empty() && args[0].type == Value::STRING) {
        std::string pattern_id = args[0].get<std::string>();
        auto it = pattern_store.find(pattern_id + "_entropy");
        if (it != pattern_store.end()) {
            return it->second;
        }
    }
    
    // Default mock entropy
    std::uniform_real_distribution<double> dist(0.2, 0.8);
    return Value(dist(rng));
}

// ============================================================================
// Quantum Operations
// ============================================================================

Value qfh_analyze(const std::vector<Value>& args) {
    std::cout << "Executing QFH analysis..." << std::endl;
    
    // Mock QFH - in real implementation, call QFHBasedProcessor::analyze()
    std::uniform_real_distribution<double> dist(0.7, 0.95);
    return Value(dist(rng));
}

Value qbsa_analyze(const std::vector<Value>& args) {
    std::cout << "Executing QBSA analysis..." << std::endl;
    
    // Mock QBSA - in real implementation, call QBSAProcessor::analyze()
    std::uniform_real_distribution<double> dist(0.6, 0.9);
    return Value(dist(rng));
}

Value manifold_optimize(const std::vector<Value>& args) {
    std::cout << "Optimizing manifold with constraints..." << std::endl;
    
    // Mock optimization - in real implementation, call QuantumManifoldOptimizationEngine
    return args.empty() ? Value("optimized_pattern") : args[0];
}

Value detect_collapse(const std::vector<Value>& args) {
    std::cout << "Detecting pattern collapse..." << std::endl;
    
    // Mock collapse detection
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return Value(dist(rng) < 0.1); // 10% chance of collapse
}

// ============================================================================
// Memory Operations
// ============================================================================

Value store_pattern(const std::vector<Value>& args) {
    if (args.size() < 2) {
        throw std::runtime_error("store_pattern requires pattern and tier arguments");
    }
    
    std::cout << "Storing pattern in memory tier..." << std::endl;
    
    // Mock storage - in real implementation, call MemoryTierManager
    static int next_id = 1000;
    return Value(std::to_string(next_id++));
}

Value retrieve_pattern(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("retrieve_pattern requires pattern ID");
    }
    
    std::cout << "Retrieving pattern by ID..." << std::endl;
    
    // Mock retrieval
    return create_pattern({});
}

Value promote_pattern(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("promote_pattern requires pattern argument");
    }
    
    std::cout << "Promoting pattern to higher tier..." << std::endl;
    
    // Mock promotion
    return Value("LTM"); // Long-term memory
}

Value query_patterns(const std::vector<Value>& args) {
    std::cout << "Querying patterns by criteria..." << std::endl;
    
    // Mock query - return pattern count
    return Value(static_cast<double>(pattern_store.size() / 3)); // Patterns have 3 metrics each
}

// ============================================================================
// Stream Operations
// ============================================================================

Value extract_bits(const std::vector<Value>& args) {
    std::cout << "Extracting bits from data stream..." << std::endl;
    
    // Mock bit extraction
    return Value("101010110101"); // Mock bitstring
}

Value weighted_sum(const std::vector<Value>& args) {
    std::cout << "Computing weighted sum..." << std::endl;
    
    // For now, return a mock weighted sum
    // In real implementation, this would take a map of weights
    std::uniform_real_distribution<double> dist(0.5, 1.0);
    return Value(dist(rng));
}

Value generate_sine_wave(const std::vector<Value>& args) {
    std::cout << "Generating sine wave..." << std::endl;
    
    double frequency = 10.0; // Default 10Hz
    if (!args.empty() && args[0].type == Value::NUMBER) {
        frequency = args[0].get<double>();
    }
    
    // Mock sine wave generation
    std::vector<double> wave;
    for (int i = 0; i < 100; ++i) {
        wave.push_back(std::sin(2 * M_PI * frequency * i / 100.0));
    }
    
    return Value("sine_wave_data"); // Mock return
}

// ============================================================================
// Type Checking & Conversion Functions (from TASK.md Phase 2A)
// ============================================================================

Value is_number(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("is_number() requires exactly 1 argument");
    }
    return Value(args[0].type == Value::NUMBER);
}

Value is_string(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("is_string() requires exactly 1 argument");
    }
    return Value(args[0].type == Value::STRING);
}

Value is_bool(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("is_bool() requires exactly 1 argument");
    }
    return Value(args[0].type == Value::BOOLEAN);
}

Value to_string(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("to_string() requires exactly 1 argument");
    }
    
    const Value& val = args[0];
    switch (val.type) {
        case Value::NUMBER:
            return Value(std::to_string(val.get<double>()));
        case Value::STRING:
            return val; // Already a string
        case Value::BOOLEAN:
            return Value(val.get<bool>() ? "true" : "false");
        case Value::PATTERN:
            return Value("pattern:" + val.get<std::string>());
        case Value::STREAM:
            return Value("stream:" + val.get<std::string>());
        default:
            return Value("unknown");
    }
}

Value to_number(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("to_number() requires exactly 1 argument");
    }
    
    const Value& val = args[0];
    switch (val.type) {
        case Value::NUMBER:
            return val; // Already a number
        case Value::STRING: {
            std::string str = val.get<std::string>();
            try {
                return Value(std::stod(str));
            } catch (const std::exception&) {
                throw std::runtime_error("Cannot convert string '" + str + "' to number");
            }
        }
        case Value::BOOLEAN:
            return Value(val.get<bool>() ? 1.0 : 0.0);
        default:
            throw std::runtime_error("Cannot convert this type to number");
    }
}

// ============================================================================
// Registration Function
// ============================================================================

void register_core_primitives(Context& context) {
    // Type checking & conversion functions
    context.set_function("is_number", is_number);
    context.set_function("is_string", is_string);
    context.set_function("is_bool", is_bool);
    context.set_function("to_string", to_string);
    context.set_function("to_number", to_number);
    
    // Pattern operations
    context.set_function("create_pattern", create_pattern);
    context.set_function("evolve_pattern", evolve_pattern);
    context.set_function("merge_patterns", merge_patterns);
    context.set_function("measure_coherence", measure_coherence);
    context.set_function("measure_stability", measure_stability);
    context.set_function("measure_entropy", measure_entropy);
    
    // Quantum operations
    context.set_function("qfh_analyze", qfh_analyze);
    context.set_function("qbsa_analyze", qbsa_analyze);
    context.set_function("manifold_optimize", manifold_optimize);
    context.set_function("detect_collapse", detect_collapse);
    
    // Memory operations
    context.set_function("store_pattern", store_pattern);
    context.set_function("retrieve_pattern", retrieve_pattern);
    context.set_function("promote_pattern", promote_pattern);
    context.set_function("query_patterns", query_patterns);
    
    // Stream operations
    context.set_function("extract_bits", extract_bits);
    context.set_function("generate_sine_wave", generate_sine_wave);
    
    // Override the weighted_sum function
    context.set_function("weighted_sum", weighted_sum);
    
    std::cout << "Registered " << 20 << " core primitive functions" << std::endl;
}

} // namespace dsl::stdlib
