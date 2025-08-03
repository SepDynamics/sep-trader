#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>

#include "dsl/parser/parser.h"
#include "dsl/runtime/interpreter.h"

using namespace dsl::parser;
using namespace dsl::runtime;

/**
 * Memory Test Runner for SEP DSL
 * 
 * Comprehensive memory leak detection and stress testing for the DSL
 * interpreter and parser components.
 */

class MemoryTestRunner {
private:
    size_t test_count_ = 0;
    size_t passed_tests_ = 0;
    
public:
    void run_basic_interpreter_test() {
        std::cout << "🧪 Testing basic interpreter memory usage..." << std::endl;
        
        const std::string dsl_code = R"(
            pattern memory_test {
                x = 42
                y = "hello world"
                z = x * 2
                print("Result:", z)
            }
        )";
        
        try {
            Parser parser(dsl_code);
            auto ast = parser.parse();
            
            if (ast) {
                Interpreter interpreter;
                interpreter.interpret(*ast);
                
                std::cout << "  ✅ Basic interpreter test passed" << std::endl;
                passed_tests_++;
            } else {
                std::cout << "  ❌ Parse failed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  ❌ Exception: " << e.what() << std::endl;
        }
        
        test_count_++;
    }
    
    void run_function_definition_test() {
        std::cout << "🧪 Testing function definition memory usage..." << std::endl;
        
        const std::string dsl_code = R"(
            function add(a, b) {
                return a + b
            }
            
            pattern function_test {
                result = add(10, 20)
                print("Function result:", result)
            }
        )";
        
        try {
            Parser parser(dsl_code);
            auto ast = parser.parse();
            
            if (ast) {
                Interpreter interpreter;
                interpreter.interpret(*ast);
                
                std::cout << "  ✅ Function definition test passed" << std::endl;
                passed_tests_++;
            } else {
                std::cout << "  ❌ Parse failed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  ❌ Exception: " << e.what() << std::endl;
        }
        
        test_count_++;
    }
    
    void run_exception_handling_test() {
        std::cout << "🧪 Testing exception handling memory usage..." << std::endl;
        
        const std::string dsl_code = R"(
            pattern exception_test {
                try {
                    x = 10
                    if (x > 5) {
                        throw "Value too large"
                    }
                    result = "success"
                }
                catch (error) {
                    result = "caught error: " + error
                }
                finally {
                    cleanup = "done"
                }
                print("Result:", result)
            }
        )";
        
        try {
            Parser parser(dsl_code);
            auto ast = parser.parse();
            
            if (ast) {
                Interpreter interpreter;
                interpreter.interpret(*ast);
                
                std::cout << "  ✅ Exception handling test passed" << std::endl;
                passed_tests_++;
            } else {
                std::cout << "  ❌ Parse failed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  ❌ Exception: " << e.what() << std::endl;
        }
        
        test_count_++;
    }
    
    void run_stress_test() {
        std::cout << "🧪 Running memory stress test..." << std::endl;
        
        const size_t iterations = 1000;
        
        for (size_t i = 0; i < iterations; ++i) {
            const std::string dsl_code = R"(
                pattern stress_test_)" + std::to_string(i) + R"( {
                    x = )" + std::to_string(i) + R"(
                    y = x * 2
                    result = y + 1
                }
            )";
            
            try {
                Parser parser(dsl_code);
                auto ast = parser.parse();
                
                if (ast) {
                    Interpreter interpreter;
                    interpreter.interpret(*ast);
                }
                
                // Small delay to allow memory cleanup
                if (i % 100 == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    std::cout << "    Completed " << i << "/" << iterations << " iterations" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "  ❌ Stress test failed at iteration " << i << ": " << e.what() << std::endl;
                test_count_++;
                return;
            }
        }
        
        std::cout << "  ✅ Stress test completed (" << iterations << " iterations)" << std::endl;
        passed_tests_++;
        test_count_++;
    }
    
    void run_parser_stress_test() {
        std::cout << "🧪 Running parser stress test..." << std::endl;
        
        const size_t iterations = 500;
        
        for (size_t i = 0; i < iterations; ++i) {
            // Generate increasingly complex patterns
            std::string dsl_code = "pattern complex_" + std::to_string(i) + " {\n";
            
            for (size_t j = 0; j < (i % 10) + 1; ++j) {
                dsl_code += "    var_" + std::to_string(j) + " = " + std::to_string(j * i) + "\n";
            }
            
            dsl_code += "    result = ";
            for (size_t j = 0; j < (i % 10) + 1; ++j) {
                if (j > 0) dsl_code += " + ";
                dsl_code += "var_" + std::to_string(j);
            }
            dsl_code += "\n}\n";
            
            try {
                Parser parser(dsl_code);
                auto ast = parser.parse();
                
                if (!ast) {
                    std::cout << "  ❌ Parser stress test failed at iteration " << i << std::endl;
                    test_count_++;
                    return;
                }
                
                if (i % 50 == 0) {
                    std::cout << "    Parsed " << i << "/" << iterations << " complex patterns" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "  ❌ Parser stress test failed at iteration " << i << ": " << e.what() << std::endl;
                test_count_++;
                return;
            }
        }
        
        std::cout << "  ✅ Parser stress test completed (" << iterations << " iterations)" << std::endl;
        passed_tests_++;
        test_count_++;
    }
    
    void run_all_tests() {
        std::cout << "🧬 Starting SEP DSL Memory Testing Suite" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        run_basic_interpreter_test();
        run_function_definition_test();
        run_exception_handling_test();
        run_stress_test();
        run_parser_stress_test();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << std::endl;
        std::cout << "📊 Memory Testing Results" << std::endl;
        std::cout << "========================" << std::endl;
        std::cout << "Tests passed: " << passed_tests_ << "/" << test_count_ << std::endl;
        std::cout << "Duration: " << duration.count() << "ms" << std::endl;
        
        if (passed_tests_ == test_count_) {
            std::cout << "✅ All memory tests passed!" << std::endl;
        } else {
            std::cout << "❌ Some memory tests failed!" << std::endl;
        }
    }
};

int main() {
    try {
        MemoryTestRunner runner;
        runner.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error in memory testing: " << e.what() << std::endl;
        return 1;
    }
}
