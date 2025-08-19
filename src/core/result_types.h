#pragma once

#include <variant>
#include <optional>
#include <string>
#include "core/quantum_types.h"
#include "io/qdrant_connector.h"

namespace sep {

// Type aliases for backward compatibility
using SEPResult = connectors::SEPResult;

// Error struct definition
struct Error {
    enum class Code {
        Success = 0,
        InvalidArgument = 1,
        NotFound = 2,
        ProcessingError = 3,
        InternalError = 4,
        NotInitialized = 5,
        CudaError = 6,
        UnknownError = 7,
        ResourceUnavailable = 8,
        OperationFailed = 9,
        NotImplemented = 10,
        AlreadyExists = 11,
        Internal = 4  // Alias for InternalError
    };
    
    Code code = Code::Success;
    std::string message;
    std::string location;
    
    Error() = default;
    Error(Code c, const std::string& msg = "") : code(c), message(msg) {}
    
    // Constructor for SEPResult compatibility
    Error(SEPResult sep_code, const std::string& msg, const std::string& loc = "") : message(msg), location(loc) {
        // Map SEPResult to Error::Code
        switch(sep_code) {
            case SEPResult::SUCCESS: code = Code::Success; break;
            case SEPResult::INVALID_ARGUMENT: code = Code::InvalidArgument; break;
            case SEPResult::NOT_FOUND: code = Code::NotFound; break;
            case SEPResult::ALREADY_EXISTS: code = Code::ProcessingError; break;
            case SEPResult::PROCESSING_ERROR: code = Code::ProcessingError; break;
            case SEPResult::NOT_INITIALIZED: code = Code::NotInitialized; break;
            case SEPResult::CUDA_ERROR: code = Code::CudaError; break;
            default: code = Code::UnknownError; break;
        }
    }
};

// Result template class
template<typename T>
class Result {
private:
    std::variant<T, Error> data_;
    
public:
    Result(const T& value) : data_(value) {}
    Result(T&& value) : data_(std::move(value)) {}
    Result(const Error& error) : data_(error) {}
    Result(Error&& error) : data_(std::move(error)) {}
    
    bool isSuccess() const { return std::holds_alternative<T>(data_); }
    bool isError() const { return std::holds_alternative<Error>(data_); }
    
    const T& value() const { return std::get<T>(data_); }
    T& value() { return std::get<T>(data_); }
    
    const Error& error() const { return std::get<Error>(data_); }
    Error& error() { return std::get<Error>(data_); }
};

// Specialization for void
template<>
class Result<void> {
private:
    std::optional<Error> error_;
    
public:
    Result() = default;
    Result(const Error& error) : error_(error) {}
    Result(Error&& error) : error_(std::move(error)) {}
    
    bool isSuccess() const { return !error_.has_value(); }
    bool isError() const { return error_.has_value(); }
    
    const Error& error() const { return error_.value(); }
    Error& error() { return error_.value(); }
    
    // Comparison operators
    bool operator==(const Result<void>& other) const {
        return error_.has_value() == other.error_.has_value() &&
               (!error_.has_value() || error_.value() == other.error_.value());
    }
    
    bool operator!=(const Result<void>& other) const {
        return !(*this == other);
    }
};

template<typename T>
Result<T> makeSuccess(T&& value) {
    return Result<T>(std::forward<T>(value));
}

template<typename T>
Result<T> makeError(const Error& error) {
    return Result<T>(error);
}

// Overload to create Error from Code and message
template<typename T>
Result<T> makeError(Error::Code code, const std::string& message) {
    return Result<T>(Error(code, message));
}

inline Result<void> makeSuccess() {
    return Result<void>();
}

inline Result<void> makeError(const Error& error) {
    return Result<void>(error);
}

// Overload to create void Result with Error from Code and message
inline Result<void> makeError(Error::Code code, const std::string& message) {
    return Result<void>(Error(code, message));
}

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