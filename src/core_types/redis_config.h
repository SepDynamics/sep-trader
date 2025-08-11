#pragma once

#include <string>

namespace sep {
namespace core_types {

struct RedisConfig {
    std::string host = "localhost";
    int port = 6379;
    std::string password = "";
};

}  // namespace core_types
}  // namespace sep
