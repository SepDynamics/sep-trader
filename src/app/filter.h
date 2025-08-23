#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace sep {
namespace services {

struct Row {
    std::string pair;
    std::chrono::time_point<std::chrono::system_clock> ts;
    double score;
    std::string id;
};

struct Filter {
    std::vector<std::function<bool(const Row&)>> predicates;
};

Filter parse_filter(std::string_view expr);
bool match(const Row& row, const Filter& filter);

} // namespace services
} // namespace sep
