#include "dsl/runtime/runtime.h"
#include <iostream>
#include <string>

void run_test(dsl::runtime::DSLRuntime& runtime, const std::string& test_name, const std::string& script) {
    std::cout << "\n--- " << test_name << " ---" << std::endl;
    try {
        runtime.execute(script);
        std::cout << "✓ " << test_name << " PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ " << test_name << " FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "DSL Test Program" << std::endl;
    std::cout << "=================" << std::endl;
    
    dsl::runtime::DSLRuntime runtime;

    // Test 1: Variable assignment and binary operations
    run_test(runtime, "Test 1: Assignment & Arithmetic", R"(
        pattern test_arithmetic {
            x = 5
            y = 10
            result = x + y
        }
    )");

    // Test 2: Function calls
    run_test(runtime, "Test 2: Function Calls", R"(
        pattern test_functions {
            c = coherence()
            s = stability()
        }
    )");

    // Test 3: Combination and signals
    run_test(runtime, "Test 3: Combination & Signals", R"(
        pattern forex_coherence {
            combined_score = coherence() * 0.7 + stability() * 0.3
        }
        
        signal buy_signal {
            trigger: forex_coherence.combined_score > 0.8
            action: BUY
        }
    )");

    std::cout << "\n--- All Tests Completed ---" << std::endl;
    return 0;
}
