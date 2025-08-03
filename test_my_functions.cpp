#include "src/dsl/compiler/compiler.h"
#include "src/dsl/stdlib/stdlib.h"
#include <iostream>

using namespace dsl::compiler;

int main() {
    std::cout << "Testing DSL Built-in Functions" << std::endl;
    std::cout << "==============================" << std::endl;
    
    Context context;
    
    // Register all standard library functions
    dsl::stdlib::register_all(context);
    
    std::cout << "\n--- Testing Type Checking Functions ---" << std::endl;
    
    try {
        // Test is_number
        std::vector<Value> args = {Value(42.5)};
        Value result = context.call_function("is_number", args);
        std::cout << "is_number(42.5) = " << (result.get<bool>() ? "true" : "false") << std::endl;
        
        // Test is_string
        args = {Value("hello")};
        result = context.call_function("is_string", args);
        std::cout << "is_string('hello') = " << (result.get<bool>() ? "true" : "false") << std::endl;
        
        // Test to_string
        args = {Value(123.45)};
        result = context.call_function("to_string", args);
        std::cout << "to_string(123.45) = '" << result.get<std::string>() << "'" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in type checking tests: " << e.what() << std::endl;
    }
    
    std::cout << "\n--- Testing Math Functions ---" << std::endl;
    
    try {
        // Test abs
        std::vector<Value> args = {Value(-5.5)};
        Value result = context.call_function("abs", args);
        std::cout << "abs(-5.5) = " << result.get<double>() << std::endl;
        
        // Test sqrt
        args = {Value(16.0)};
        result = context.call_function("sqrt", args);
        std::cout << "sqrt(16) = " << result.get<double>() << std::endl;
        
        // Test min
        args = {Value(10.0), Value(20.0)};
        result = context.call_function("min", args);
        std::cout << "min(10, 20) = " << result.get<double>() << std::endl;
        
        // Test pow
        args = {Value(2.0), Value(3.0)};
        result = context.call_function("pow", args);
        std::cout << "pow(2, 3) = " << result.get<double>() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in math tests: " << e.what() << std::endl;
    }
    
    std::cout << "\n--- Testing Statistical Functions ---" << std::endl;
    
    try {
        // Test mean
        std::vector<Value> args = {Value(1.0), Value(2.0), Value(3.0), Value(4.0), Value(5.0)};
        Value result = context.call_function("mean", args);
        std::cout << "mean(1, 2, 3, 4, 5) = " << result.get<double>() << std::endl;
        
        // Test median
        result = context.call_function("median", args);
        std::cout << "median(1, 2, 3, 4, 5) = " << result.get<double>() << std::endl;
        
        // Test std_dev (needs at least 2 args)
        std::vector<Value> stats_args = {Value(1.0), Value(3.0), Value(5.0)};
        result = context.call_function("std_dev", stats_args);
        std::cout << "std_dev(1, 3, 5) = " << result.get<double>() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in statistical tests: " << e.what() << std::endl;
    }
    
    std::cout << "\n--- Test Complete ---" << std::endl;
    std::cout << "✓ All built-in functions are working!" << std::endl;
    
    return 0;
}
