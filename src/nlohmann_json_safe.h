#pragma once

#include <array>

// This is a centralized, safe include for nlohmann::json.
// It ensures that we use a consistent configuration for the library across the entire project.

// Disable exceptions for performance and to avoid issues with CUDA.
#define NLOHMANN_JSON_NOEXCEPTION 1

// Include the nlohmann/json header.
#include <nlohmann/json.hpp>

// Add any other safety features or using declarations here.
