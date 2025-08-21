#pragma once

// This file is deprecated and unused - services::Result is not referenced anywhere
// Keeping minimal header to avoid breaking any potential legacy includes
// All actual Result usage should reference sep::Result from result_types.h

#include "result_types.h"

namespace sep {
namespace services {
    // Forward declare deprecated types for legacy compatibility only
    // DO NOT USE - Use sep::Result and sep::Error instead
    template<typename T> class Result;
    class Error;
} // namespace services
} // namespace sep