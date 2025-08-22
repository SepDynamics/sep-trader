#include "math.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace dsl::stdlib {

// ============================================================================
// Basic Arithmetic Functions (TASK.md Phase 2A Priority 1)
// ============================================================================

Value abs_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("abs() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("abs() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::abs(x));
}

Value min_func(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw std::runtime_error("min() expects exactly 2 arguments");
    }
    if (!std::holds_alternative<double>(args[0]) || !std::holds_alternative<double>(args[1])) {
        throw std::runtime_error("min() requires number arguments");
    }
    double a = std::get<double>(args[0]);
    double b = std::get<double>(args[1]);
    return Value(std::min(a, b));
}

Value max_func(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw std::runtime_error("max() expects exactly 2 arguments");
    }
    if (!std::holds_alternative<double>(args[0]) || !std::holds_alternative<double>(args[1])) {
        throw std::runtime_error("max() requires number arguments");
    }
    double a = std::get<double>(args[0]);
    double b = std::get<double>(args[1]);
    return Value(std::max(a, b));
}

Value round_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("round() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("round() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::round(x));
}

Value floor_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("floor() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("floor() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::floor(x));
}

Value ceil_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("ceil() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("ceil() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::ceil(x));
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

Value sin_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("sin() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("sin() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::sin(x));
}

Value cos_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("cos() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("cos() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::cos(x));
}

Value tan_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("tan() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("tan() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::tan(x));
}

Value asin_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("asin() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("asin() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    if (x < -1.0 || x > 1.0) {
        throw std::runtime_error("asin() domain error: argument must be in [-1, 1]");
    }
    return Value(std::asin(x));
}

Value acos_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("acos() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("acos() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    if (x < -1.0 || x > 1.0) {
        throw std::runtime_error("acos() domain error: argument must be in [-1, 1]");
    }
    return Value(std::acos(x));
}

Value atan_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("atan() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("atan() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::atan(x));
}

// ============================================================================
// Exponential/Logarithmic Functions
// ============================================================================

Value exp_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("exp() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("exp() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    return Value(std::exp(x));
}

Value log_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("log() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("log() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    if (x <= 0.0) {
        throw std::runtime_error("log() domain error: argument must be positive");
    }
    return Value(std::log(x));
}

Value log10_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("log10() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("log10() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    if (x <= 0.0) {
        throw std::runtime_error("log10() domain error: argument must be positive");
    }
    return Value(std::log10(x));
}

Value pow_func(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw std::runtime_error("pow() expects exactly 2 arguments");
    }
    if (!std::holds_alternative<double>(args[0]) || !std::holds_alternative<double>(args[1])) {
        throw std::runtime_error("pow() requires number arguments");
    }
    double x = std::get<double>(args[0]);
    double y = std::get<double>(args[1]);
    return Value(std::pow(x, y));
}

Value sqrt_func(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw std::runtime_error("sqrt() expects exactly 1 argument");
    }
    if (!std::holds_alternative<double>(args[0])) {
        throw std::runtime_error("sqrt() requires a number argument");
    }
    double x = std::get<double>(args[0]);
    if (x < 0.0) {
        throw std::runtime_error("sqrt() domain error: argument must be non-negative");
    }
    return Value(std::sqrt(x));
}

// ============================================================================
// Registration Function
// ============================================================================

void register_math(Context& context) {
    // Basic arithmetic functions
    context.set_function("abs", abs_func);
    context.set_function("min", min_func);
    context.set_function("max", max_func);
    context.set_function("round", round_func);
    context.set_function("floor", floor_func);
    context.set_function("ceil", ceil_func);
    
    // Trigonometric functions
    context.set_function("sin", sin_func);
    context.set_function("cos", cos_func);
    context.set_function("tan", tan_func);
    context.set_function("asin", asin_func);
    context.set_function("acos", acos_func);
    context.set_function("atan", atan_func);
    
    // Exponential/logarithmic functions
    context.set_function("exp", exp_func);
    context.set_function("log", log_func);
    context.set_function("log10", log10_func);
    context.set_function("pow", pow_func);
    context.set_function("sqrt", sqrt_func);
}

}
