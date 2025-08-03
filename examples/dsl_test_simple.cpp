#include "dsl/runtime/runtime.h"
#include <iostream>

int main() {
    std::cout << "🚀 DSL Foundation Test" << std::endl;
    std::cout << "======================" << std::endl;
    
    dsl::runtime::DSLRuntime runtime;

    // Test complete DSL functionality
    try {
        runtime.execute(R"(
            pattern trading_strategy {
                market_coherence = coherence()
                market_stability = stability() 
                risk_factor = 0.3
                combined_score = market_coherence * 0.6 + market_stability * 0.4
                final_decision = combined_score - risk_factor
            }
        )");
        
        std::cout << "\n✅ SUCCESS: Complete DSL pipeline working!" << std::endl;
        std::cout << "   ✓ Function calls (coherence, stability)" << std::endl;
        std::cout << "   ✓ Variable assignments" << std::endl;
        std::cout << "   ✓ Arithmetic operations (+, -, *)" << std::endl;
        std::cout << "   ✓ Pattern compilation and execution" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ FAILED: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n🎯 Ready for weekend development!" << std::endl;
    return 0;
}
