#pragma once

#include <string>

namespace sep {
namespace core_types {

struct RedisConfig {
    std::string host = std::getenv("SEP_CACHE_HOST") ? std::getenv("SEP_CACHE_HOST") : "localhost";
    int port = std::getenv("SEP_CACHE_PORT") ? std::atoi(std::getenv("SEP_CACHE_PORT")) : 6379;
    std::string password = std::getenv("SEP_CACHE_PASSWORD") ? std::getenv("SEP_CACHE_PASSWORD") : "";
};

}  // namespace core_types
}  // namespace sep
