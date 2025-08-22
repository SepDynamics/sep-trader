#pragma once
// Crow HTTP request wrapper

#include "crow.h"

// Re-export Crow request types
using crow_request = ::crow::request;
using crow_response = ::crow::response;
using crow_method = ::crow::HTTPMethod;
