#include <sep_precompiled.h>
#include <functional>
#include <iostream>

int main() {
    std::array<int, 5> test_array = {1, 2, 3, 4, 5};
    std::cout << "std::array works: " << test_array[0] << std::endl;
    return 0;
}
