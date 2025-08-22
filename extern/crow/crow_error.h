#pragma once
// Crow error handling utilities

#include <string>

namespace sep {
namespace crow {
namespace error {

enum class Code {
    NONE = 0,
    INVALID_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500
};

void log(Code code, const std::string& message);
void set_last_error(const std::string& error);
const char* get_last_error();

} // namespace error
} // namespace crow
} // namespace sep

namespace crow {
    extern const char* last_error;
}
