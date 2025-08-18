#pragma once

#include "core/quantum_types.h"
#include "io/qdrant_connector.h"

namespace sep {

// Type aliases for backward compatibility
using SEPResult = connectors::SEPResult;

// Forward declarations to avoid circular dependencies
template<typename T> class Result;
struct Error;

template<typename T>
Result<T> makeSuccess(T&& value);

template<typename T>
Result<T> makeError(const Error& error);

// Additional result utilities for enum-based results
namespace result {
    constexpr SEPResult SUCCESS = SEPResult::SUCCESS;
    constexpr SEPResult ERROR = SEPResult::PROCESSING_ERROR;
    constexpr SEPResult INVALID_ARGUMENT = SEPResult::INVALID_ARGUMENT;
    constexpr SEPResult NOT_FOUND = SEPResult::NOT_FOUND;
}

// Namespace aliases for backward compatibility
namespace core {
    // Template Result<T> for new code
    template<typename T>
    using Result = sep::Result<T>;
    
    // Import quantum types
    using Pattern = sep::quantum::Pattern;
    using QuantumState = sep::quantum::QuantumState;
    using PatternRelationship = sep::quantum::PatternRelationship;
    using RelationshipType = sep::quantum::RelationshipType;
    
    // POD types for conversions
    using QuantumStatePOD = sep::quantum::QuantumStatePOD;
    using PatternRelationshipPOD = sep::quantum::PatternRelationshipPOD;
    using PatternPOD = sep::quantum::PatternPOD;
}

} // namespace sep