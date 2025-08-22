// Simple test to verify CUDA+JSON compatibility

#include "common/sep_precompiled.h"
#include "nlohmann_json_safe.h"

int main() {
    nlohmann::json j;
    j["hello"] = "world";
    
    // Test std::array with JSON
    std::array<int, 3> arr = {1, 2, 3};
    j["array"] = arr;
    
    return 0;
}
