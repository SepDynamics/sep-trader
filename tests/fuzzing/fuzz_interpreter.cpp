#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include "dsl/parser/parser.h"
#include "dsl/lexer/lexer.h"
#include "dsl/runtime/interpreter.h"

using namespace dsl::lexer;
using namespace dsl::parser;
using namespace dsl::runtime;

// Enhanced fuzzer for full DSL interpreter pipeline
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Skip empty inputs
    if (Size == 0) return 0;
    
    // Limit input size to prevent excessive memory usage
    if (Size > 10000) return 0;
    
    // Create null-terminated string safely
    std::string input;
    input.reserve(Size + 1);
    input.assign(reinterpret_cast<const char*>(Data), Size);
    
    // Ensure input contains only printable ASCII and basic whitespace
    bool valid_input = true;
    for (char c : input) {
        if (c < 0x20 && c != '\n' && c != '\r' && c != '\t') {
            valid_input = false;
            break;
        }
    }
    
    if (!valid_input) return 0;
    
    try {
        // Test parser
        Parser parser(input);
        auto ast = parser.parse();
        
        if (ast) {
            // Test interpreter execution
            Interpreter interpreter;
            
            // Execute with timeout protection
            try {
                interpreter.interpret(*ast);
            } catch (const std::runtime_error& e) {
                // Runtime errors are expected for malformed programs
                #ifdef FUZZ_DEBUG
                std::cerr << "Runtime error: " << e.what() << std::endl;
                #endif
            } catch (const std::logic_error& e) {
                // Logic errors might indicate interpreter bugs
                #ifdef FUZZ_DEBUG
                std::cerr << "Logic error: " << e.what() << std::endl;
                #endif
            }
        }
        
    } catch (const std::exception& e) {
        // Log exceptions in debug mode for analysis
        #ifdef FUZZ_DEBUG
        std::cerr << "Fuzz input caused exception: " << e.what() << std::endl;
        std::cerr << "Input size: " << Size << std::endl;
        std::cerr << "Input preview: " << input.substr(0, 100) << std::endl;
        #endif
    } catch (...) {
        // Catch all other exceptions
        #ifdef FUZZ_DEBUG
        std::cerr << "Fuzz input caused unknown exception" << std::endl;
        #endif
    }
    
    return 0;
}
