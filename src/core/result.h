#pragma once

// Redirect to canonical Result implementation in result_types.h
// This eliminates the namespace conflict between sep::core::Result and sep::Result
#include "result_types.h"

namespace sep {
namespace core {
    // Alias to canonical Result implementation
    template<typename T>
    using Result = sep::Result<T>;
    
    // Alias error type for backward compatibility
    using Error = sep::Error;
} // namespace core
} // namespace sep