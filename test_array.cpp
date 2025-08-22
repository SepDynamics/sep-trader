// test_array.cpp - Diagnostic test for std::array template issues
#include <iostream>
#include <array>
#include <functional>  // To trigger the functional header issue
#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <cstddef>
#include <cstdint>

int main() {
    // Test basic std::array usage
    std::array<int, 5> my_array;
    my_array.fill(42);
    std::cout << "Array element: " << my_array[0] << std::endl;

    // Test std::atomic<std::chrono::time_point>
    std::atomic<std::chrono::system_clock::time_point> atomic_time{std::chrono::system_clock::now()};
    std::cout << "Atomic time: " << atomic_time.load().time_since_epoch().count() << std::endl;

    // Test optional
    std::optional<int> opt_val = 10;
    if (opt_val) {
        std::cout << "Optional value: " << *opt_val << std::endl;
    }

    // Test functional usage that might trigger the error
    std::function<void()> f = []() { std::cout << "Lambda works" << std::endl; };
    f();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
