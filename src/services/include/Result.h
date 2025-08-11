#pragma once

#include <variant>
#include <string>
#include <optional>
#include <type_traits>
#include <stdexcept>  // Added for std::runtime_error

namespace sep {
namespace services {

/**
 * Generic error class for service operations
 */
class Error {
public:
    enum class Code {
        None = 0,
        InvalidArgument,
        NotFound,
        AccessDenied,
        ResourceUnavailable,
        OperationFailed,
        Timeout,
        Internal
    };

    Error() : m_code(Code::None) {}
    Error(Code code, const std::string& message) : m_code(code), m_message(message) {}

    Code code() const { return m_code; }
    const std::string& message() const { return m_message; }

    bool isError() const { return m_code != Code::None; }
    operator bool() const { return isError(); }

private:
    Code m_code;
    std::string m_message;
};

/**
 * Generic result type for service operations
 * Contains either a value of type T or an error
 */
template<typename T>
class Result {
public:
    Result() : m_variant(Error()) {}
    Result(const T& value) : m_variant(value) {}
    Result(T&& value) : m_variant(std::move(value)) {}
    Result(const Error& error) : m_variant(error) {}

    bool isError() const { 
        return std::holds_alternative<Error>(m_variant) && std::get<Error>(m_variant).isError(); 
    }
    
    bool hasValue() const { return std::holds_alternative<T>(m_variant); }
    
    const T& value() const { 
        if (!hasValue()) {
            throw std::runtime_error("Result does not contain a value");
        }
        return std::get<T>(m_variant); 
    }
    
    T& value() { 
        if (!hasValue()) {
            throw std::runtime_error("Result does not contain a value");
        }
        return std::get<T>(m_variant); 
    }
    
    const Error& error() const {
        if (!isError()) {
            static const Error none;
            return none;
        }
        return std::get<Error>(m_variant);
    }

    /**
     * Apply a function to the contained value if present
     * @param func Function to apply to the value
     * @return Result containing the function result or the original error
     */
    template<typename Func>
    auto map(Func&& func) const -> Result<decltype(func(std::declval<T>()))> {
        using ReturnType = decltype(func(std::declval<T>()));
        
        if (isError()) {
            return Result<ReturnType>(error());
        }
        
        return Result<ReturnType>(func(value()));
    }
    
    /**
     * Chain operations that might fail
     * @param func Function that returns a Result
     * @return Result from the function or the original error
     */
    template<typename Func>
    auto flatMap(Func&& func) const -> decltype(func(std::declval<T>())) {
        using ReturnType = decltype(func(std::declval<T>()));
        
        if (isError()) {
            return ReturnType(error());
        }
        
        return func(value());
    }

private:
    std::variant<T, Error> m_variant;
};

// Specialization for void results
template<>
class Result<void> {
public:
    Result() : m_error(std::nullopt) {}
    Result(const Error& error) : m_error(error) {}

    bool isError() const { return m_error.has_value() && m_error->isError(); }
    
    bool hasValue() const { return !isError(); }
    
    const Error& error() const {
        if (!isError()) {
            static const Error none;
            return none;
        }
        return *m_error;
    }

    /**
     * Chain operations that might fail
     * @param func Function that returns a Result
     * @return Result from the function or the original error
     */
    template<typename Func>
    auto flatMap(Func&& func) const -> decltype(func()) {
        using ReturnType = decltype(func());
        
        if (isError()) {
            return ReturnType(error());
        }
        
        return func();
    }

private:
    std::optional<Error> m_error;
};

} // namespace services
} // namespace sep