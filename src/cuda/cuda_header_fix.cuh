/**
 * @file cuda_header_fix.cuh
 * @brief A dedicated header to fix C++ standard library issues with CUDA.
 *
 * This header should be included at the top of any CUDA file (.cu) to ensure
 * proper handling of standard library headers and to avoid common compilation errors.
 */

#pragma once

//==============================================================================
// CRITICAL: C headers must come first
//==============================================================================
#include "csignal_compat.h"
#include <cstddef>
#include <cstdint>

//==============================================================================
// CRITICAL: Include <array> before any other C++ header
//==============================================================================
#include <array>

//==============================================================================
// Other required C++ Standard Library Headers
//==============================================================================
#include <vector>
#include <string>
#include <iostream>