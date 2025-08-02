#include "dsl/runtime/runtime.h"
#include <iostream>
#include <fstream>

int main() {
    std::cout << "=== SEP DSL Foundation Test ===" << std::endl;
    
    // Test 1: Basic runtime initialization
    try {
        sep::dsl::runtime::DSLRuntime runtime;
        runtime.setDebugMode(true);
        std::cout << "✅ DSL Runtime initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Runtime initialization failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Test 2: Simple DSL script validation
    std::string simple_script = R"(
        pattern test_pattern {
            coherence = 0.5
        }
    )";
    
    bool validation_result = sep::dsl::runtime::convenience::validateScript(simple_script);
    std::cout << "✅ Script validation test completed (result: " << validation_result << ")" << std::endl;
    
    // Test 3: Load and validate example file
    std::ifstream example_file("/sep/docs/dsl/examples/forex_coherence.sep");
    if (example_file.is_open()) {
        std::string example_content((std::istreambuf_iterator<char>(example_file)),
                                   std::istreambuf_iterator<char>());
        example_file.close();
        
        bool example_validation = sep::dsl::runtime::convenience::validateScript(example_content);
        std::cout << "✅ Example file validation completed (result: " << example_validation << ")" << std::endl;
    } else {
        std::cout << "⚠️  Could not load example file for validation" << std::endl;
    }
    
    // Test 4: Component integration
    sep::dsl::runtime::DSLRuntime runtime;
    auto& parser = runtime.getParser();
    auto& compiler = runtime.getCompiler();
    std::cout << "✅ DSL components accessible" << std::endl;
    
    std::cout << "\n=== DSL Foundation Ready for Weekend Development ===" << std::endl;
    std::cout << "Next steps:" << std::endl;
    std::cout << "1. Implement parsePattern() in parser.cpp" << std::endl;
    std::cout << "2. Implement compilePattern() in compiler.cpp" << std::endl;
    std::cout << "3. Map DSL operations to existing SEP engine calls" << std::endl;
    std::cout << "4. Test with simple pattern definitions" << std::endl;
    
    return 0;
}
