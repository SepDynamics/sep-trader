#pragma once
// Crow framework isolation header
// This provides a controlled interface to Crow to avoid direct dependencies

#include "crow.h"

// Re-export essential Crow types and functions
namespace crow {
    using ::crow::request;
    using ::crow::response;
    using ::crow::HTTPMethod;
    using ::crow::status;
    using ::crow::method_name;
    using ::crow::Crow;
}

// Define compatibility macros
#ifndef CROW_DISABLE_RTTI
#define CROW_DISABLE_RTTI 0
#endif
