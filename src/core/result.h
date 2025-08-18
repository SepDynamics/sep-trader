#pragma once

#include <variant>
#include <string>
#include <optional>

// Forward declaration
namespace sep {
    struct Error;
}

// Include complete Error definition to fix incomplete type errors
#include "core/error_handler.h"

namespace sep {

/**
 * Result<T> template class - can hold either a value of type T or an Error
 * Uses the existing Error struct from error_handler.h
 * Specializes for void type using std::monostate
 */
template<typename T>
class Result {
public:
    // Constructor for success case
    Result(const T& value) : data_(value) {}
    Result(T&& value) : data_(std::move(value)) {}
    
    // Constructor for error case
    Result(const Error& error) : data_(error) {}
    Result(Error&& error) : data_(std::move(error)) {}
    
    // Copy constructor
    Result(const Result& other) : data_(other.data_) {}
    
    // Move constructor
    Result(Result&& other) noexcept : data_(std::move(other.data_)) {}
    
    // Assignment operators
    Result& operator=(const Result& other) {
        if (this != &other) {
            data_ = other.data_;
        }
        return *this;
    }
    
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }
    
    // Check if this result contains an error
    bool isError() const {
        return std::holds_alternative<Error>(data_);
    }
    
    // Check if this result contains a value (opposite of isError)
    bool isSuccess() const {
        return std::holds_alternative<T>(data_);
    }
    
    // Get the error (only valid if isError() returns true)
    const Error& error() const {
        return std::get<Error>(data_);
    }
    
    // Get the value (only valid if isSuccess() returns true)
    const T& value() const {
        return std::get<T>(data_);
    }
    
    // Get the value (only valid if isSuccess() returns true)
    T& value() {
        return std::get<T>(data_);
    }
    
    // Operator bool - returns true if success, false if error
    explicit operator bool() const {
        return isSuccess();
    }
    
    // Operator* - get value (undefined behavior if error)
    const T& operator*() const {
        return value();
    }
    
    T& operator*() {
        return value();
    }
    
    // Operator-> - access value members (undefined behavior if error)
    const T* operator->() const {
        return &value();
    }
    
    T* operator->() {
        return &value();
    }

private:
    std::variant<T, Error> data_;
};

/**
 * Specialization for Result<void> - uses std::monostate to represent void success
 */
template<>
class Result<void> {
public:
    // Constructor for success case
    Result() : data_(std::monostate{}) {}
    
    // Constructor for error case
    Result(const Error& error) : data_(error) {}
    Result(Error&& error) : data_(std::move(error)) {}
    
    // Copy constructor
    Result(const Result& other) : data_(other.data_) {}
    
    // Move constructor
    Result(Result&& other) noexcept : data_(std::move(other.data_)) {}
    
    // Assignment operators
    Result& operator=(const Result& other) {
        if (this != &other) {
            data_ = other.data_;
        }
        return *this;
    }
    
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }
    
    // Check if this result contains an error
    bool isError() const {
        return std::holds_alternative<Error>(data_);
    }
    
    // Check if this result contains a value (opposite of isError)
    bool isSuccess() const {
        return std::holds_alternative<std::monostate>(data_);
    }
    
    // Get the error (only valid if isError() returns true)
    const Error& error() const {
        return std::get<Error>(data_);
    }
    
    // Operator bool - returns true if success, false if error
    explicit operator bool() const {
        return isSuccess();
    }

private:
    std::variant<std::monostate, Error> data_;
};

// Result creation helper functions
template<typename T>
Result<T> makeSuccess(T&& value) {
    return Result<T>(std::forward<T>(value));
}

// Specialization for void success
inline Result<void> makeSuccess() {
    return Result<void>();
}

template<typename T>
Result<T> makeError(const Error& error) {
    return Result<T>(error);
}

} // namespace sep