#include "dsl/runtime/runtime.h"
#include <iostream>

int main() {
    dsl::runtime::DSLRuntime runtime;
    
    std::cout << "Testing simple expression parsing..." << std::endl;
    
    try {
        // Test very simple pattern
        std::string simple = R"(
            pattern test {
                x = 5
            }
        )";
        
        runtime.execute(simple);
        std::cout << "Simple test succeeded!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
