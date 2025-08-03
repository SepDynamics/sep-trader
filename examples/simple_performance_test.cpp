#include <iostream>
#include <chrono>
#include <cmath>

using namespace std::chrono;

int main() {
    const int iterations = 100000;
    
    // C++ baseline - simple arithmetic
    auto start = high_resolution_clock::now();
    double result = 0.0;
    for (int i = 0; i < iterations; i++) {
        double x = static_cast<double>(i);
        result += x * 2 + 1; // Simple arithmetic
    }
    auto end = high_resolution_clock::now();
    auto cpp_simple = duration_cast<microseconds>(end - start);
    
    // C++ baseline - math functions
    start = high_resolution_clock::now();
    result = 0.0;
    for (int i = 0; i < iterations; i++) {
        double x = static_cast<double>(i);
        result += std::sin(x) + std::cos(x) + std::sqrt(x + 1);
    }
    end = high_resolution_clock::now();
    auto cpp_math = duration_cast<microseconds>(end - start);
    
    std::cout << "C++ Performance Baseline (100k iterations):" << std::endl;
    std::cout << "Simple arithmetic: " << cpp_simple.count() << " μs" << std::endl;
    std::cout << "Math functions: " << cpp_math.count() << " μs" << std::endl;
    std::cout << "Result: " << result << std::endl;
    
    return 0;
}
