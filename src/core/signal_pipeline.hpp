#pragma once
#include <string>
#include <vector>

namespace sep::trading {
std::string runSignalPipeline(const std::string& pair, const std::vector<double>& prices);
}
