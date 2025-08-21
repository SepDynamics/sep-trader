#pragma once

// This header now just includes the canonical result_types.h to avoid duplicate definitions
// All Result<T> functionality is defined in result_types.h

#include "core/result_types.h"

namespace sep {
    // All Result types and functions are imported from result_types.h
    // This header is kept for backward compatibility
}

// Legacy namespace compatibility
namespace core {
    using ::sep::Result;
    using ::sep::Error;
}
