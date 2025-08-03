#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include "../src/dsl/parser/parser.h"
#include "../src/dsl/runtime/interpreter.h"

using namespace std::chrono;

class PerformanceBenchmark {
public:
    // C++ native implementation of math operations
    double cpp_math_operations(int iterations) {
        double result = 0.0;
        for (int i = 0; i < iterations; i++) {
            double x = static_cast<double>(i);
            result += std::sin(x) + std::cos(x) + std::sqrt(x + 1) + std::log(x + 1);
        }
        return result;
    }
    
    // DSL implementation of the same operations
    double dsl_math_operations(int iterations) {
        std::string dsl_code = R"(
pattern benchmark_math {
    function math_ops(x: Number): Number {
        return sin(x) + cos(x) + sqrt(x + 1) + log(x + 1)
    }
    
    total: Number = 0
    i: Number = 0
    while (i < )" + std::to_string(iterations) + R"() {
        total = total + math_ops(i)
        i = i + 1
    }
}
)";
        
        try {
            dsl::parser::Parser parser(dsl_code);
            auto program = parser.parse();
            
            dsl::runtime::Interpreter interpreter;
            interpreter.interpret(*program);
            
            // Extract the result (this is a simplification)
            return 42.0; // Placeholder - in reality we'd extract the actual result
        } catch (const std::exception& e) {
            std::cerr << "DSL Error: " << e.what() << std::endl;
            return 0.0;
        }
    }
    
    // Function call overhead test
    double cpp_function_calls(int iterations) {
        double result = 0.0;
        for (int i = 0; i < iterations; i++) {
            result += simple_function(static_cast<double>(i));
        }
        return result;
    }
    
    double dsl_function_calls(int iterations) {
        std::string dsl_code = R"(
pattern benchmark_functions {
    function simple_func(x: Number): Number {
        return x * 2 + 1
    }
    
    total: Number = 0
    i: Number = 0
    while (i < )" + std::to_string(iterations) + R"() {
        total = total + simple_func(i)
        i = i + 1
    }
}
)";
        
        try {
            dsl::parser::Parser parser(dsl_code);
            auto program = parser.parse();
            
            dsl::runtime::Interpreter interpreter;
            interpreter.interpret(*program);
            
            return 42.0; // Placeholder
        } catch (const std::exception& e) {
            std::cerr << "DSL Error: " << e.what() << std::endl;
            return 0.0;
        }
    }
    
    void run_benchmarks() {
        std::vector<int> test_sizes = {1000, 10000, 100000};
        
        std::cout << "=== DSL Performance Benchmark ===" << std::endl;
        std::cout << "Comparing DSL interpreter vs native C++ performance" << std::endl;
        std::cout << std::endl;
        
        for (int size : test_sizes) {
            std::cout << "Test size: " << size << " iterations" << std::endl;
            
            // Math operations benchmark
            auto start = high_resolution_clock::now();
            double cpp_result = cpp_math_operations(size);
            auto cpp_end = high_resolution_clock::now();
            auto cpp_duration = duration_cast<microseconds>(cpp_end - start);
            
            start = high_resolution_clock::now();
            double dsl_result = dsl_math_operations(size);
            auto dsl_end = high_resolution_clock::now();
            auto dsl_duration = duration_cast<microseconds>(dsl_end - start);
            
            double overhead = static_cast<double>(dsl_duration.count()) / cpp_duration.count();
            
            std::cout << "  Math Operations:" << std::endl;
            std::cout << "    C++: " << cpp_duration.count() << " μs" << std::endl;
            std::cout << "    DSL: " << dsl_duration.count() << " μs" << std::endl;
            std::cout << "    Overhead: " << overhead << "x" << std::endl;
            
            // Function calls benchmark
            start = high_resolution_clock::now();
            cpp_result = cpp_function_calls(size);
            cpp_end = high_resolution_clock::now();
            cpp_duration = duration_cast<microseconds>(cpp_end - start);
            
            start = high_resolution_clock::now();
            dsl_result = dsl_function_calls(size);
            dsl_end = high_resolution_clock::now();
            dsl_duration = duration_cast<microseconds>(dsl_end - start);
            
            overhead = static_cast<double>(dsl_duration.count()) / cpp_duration.count();
            
            std::cout << "  Function Calls:" << std::endl;
            std::cout << "    C++: " << cpp_duration.count() << " μs" << std::endl;
            std::cout << "    DSL: " << dsl_duration.count() << " μs" << std::endl;
            std::cout << "    Overhead: " << overhead << "x" << std::endl;
            std::cout << std::endl;
        }
    }
    
private:
    double simple_function(double x) {
        return x * 2 + 1;
    }
};

int main() {
    PerformanceBenchmark benchmark;
    benchmark.run_benchmarks();
    return 0;
}
