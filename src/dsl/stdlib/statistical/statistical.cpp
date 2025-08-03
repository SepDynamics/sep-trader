#include "statistical.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dsl::stdlib {

// Helper function to extract numeric values from argument list
std::vector<double> extract_numbers(const std::vector<Value>& args) {
    std::vector<double> numbers;
    for (const auto& arg : args) {
        if (arg.type != Value::NUMBER) {
            throw std::runtime_error("Statistical functions require numeric arguments");
        }
        numbers.push_back(arg.get<double>());
    }
    return numbers;
}

// ============================================================================
// Basic Statistical Functions (TASK.md Phase 2A Priority 1)
// ============================================================================

Value mean_func(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("mean() requires at least 1 argument");
    }
    
    std::vector<double> data = extract_numbers(args);
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return Value(sum / data.size());
}

Value median_func(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("median() requires at least 1 argument");
    }
    
    std::vector<double> data = extract_numbers(args);
    std::sort(data.begin(), data.end());
    
    size_t n = data.size();
    if (n % 2 == 0) {
        // Even number of elements - average of middle two
        return Value((data[n/2 - 1] + data[n/2]) / 2.0);
    } else {
        // Odd number of elements - middle element
        return Value(data[n/2]);
    }
}

Value variance_func(const std::vector<Value>& args) {
    if (args.size() < 2) {
        throw std::runtime_error("variance() requires at least 2 arguments");
    }
    
    std::vector<double> data = extract_numbers(args);
    
    // Calculate mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    
    // Calculate sum of squared differences
    double sum_sq_diff = 0.0;
    for (double x : data) {
        double diff = x - mean;
        sum_sq_diff += diff * diff;
    }
    
    // Use sample variance (divide by n-1)
    return Value(sum_sq_diff / (data.size() - 1));
}

Value std_dev_func(const std::vector<Value>& args) {
    if (args.size() < 2) {
        throw std::runtime_error("std_dev() requires at least 2 arguments");
    }
    
    // Calculate variance first
    Value var_result = variance_func(args);
    double variance = var_result.get<double>();
    
    return Value(std::sqrt(variance));
}

Value min_value_func(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("min_value() requires at least 1 argument");
    }
    
    std::vector<double> data = extract_numbers(args);
    double min_val = *std::min_element(data.begin(), data.end());
    return Value(min_val);
}

Value max_value_func(const std::vector<Value>& args) {
    if (args.empty()) {
        throw std::runtime_error("max_value() requires at least 1 argument");
    }
    
    std::vector<double> data = extract_numbers(args);
    double max_val = *std::max_element(data.begin(), data.end());
    return Value(max_val);
}

// ============================================================================
// Advanced Statistical Functions
// ============================================================================

Value correlation_func(const std::vector<Value>& args) {
    if (args.size() % 2 != 0 || args.size() < 4) {
        throw std::runtime_error("correlation() requires an even number of arguments (at least 4) representing two equal-length datasets");
    }
    
    std::vector<double> all_data = extract_numbers(args);
    size_t n = all_data.size() / 2;
    
    std::vector<double> x(all_data.begin(), all_data.begin() + n);
    std::vector<double> y(all_data.begin() + n, all_data.end());
    
    // Calculate means
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;
    
    // Calculate correlation components
    double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    // Pearson correlation coefficient
    double denominator = std::sqrt(sum_x2 * sum_y2);
    if (denominator == 0.0) {
        throw std::runtime_error("correlation() division by zero - one dataset has no variance");
    }
    
    return Value(sum_xy / denominator);
}

Value covariance_func(const std::vector<Value>& args) {
    if (args.size() % 2 != 0 || args.size() < 4) {
        throw std::runtime_error("covariance() requires an even number of arguments (at least 4) representing two equal-length datasets");
    }
    
    std::vector<double> all_data = extract_numbers(args);
    size_t n = all_data.size() / 2;
    
    std::vector<double> x(all_data.begin(), all_data.begin() + n);
    std::vector<double> y(all_data.begin() + n, all_data.end());
    
    // Calculate means
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;
    
    // Calculate covariance
    double sum_cov = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    
    // Use sample covariance (divide by n-1)
    return Value(sum_cov / (n - 1));
}

Value percentile_func(const std::vector<Value>& args) {
    if (args.size() < 2) {
        throw std::runtime_error("percentile() requires at least 2 arguments: percentile value followed by data");
    }
    
    if (args[0].type != Value::NUMBER) {
        throw std::runtime_error("percentile() first argument must be the percentile value (0-100)");
    }
    
    double p = args[0].get<double>();
    if (p < 0.0 || p > 100.0) {
        throw std::runtime_error("percentile() value must be between 0 and 100");
    }
    
    // Extract data (skip first argument which is the percentile)
    std::vector<Value> data_args(args.begin() + 1, args.end());
    std::vector<double> data = extract_numbers(data_args);
    
    if (data.empty()) {
        throw std::runtime_error("percentile() requires at least one data point");
    }
    
    std::sort(data.begin(), data.end());
    
    // Calculate index for percentile
    double index = (p / 100.0) * (data.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));
    
    if (lower == upper) {
        return Value(data[lower]);
    } else {
        // Linear interpolation
        double weight = index - lower;
        return Value(data[lower] * (1.0 - weight) + data[upper] * weight);
    }
}

// ============================================================================
// Registration Function
// ============================================================================

void register_statistical(Context& context) {
    // Basic statistical functions
    context.set_function("mean", mean_func);
    context.set_function("median", median_func);
    context.set_function("std_dev", std_dev_func);
    context.set_function("variance", variance_func);
    context.set_function("min_value", min_value_func);
    context.set_function("max_value", max_value_func);
    
    // Advanced statistical functions
    context.set_function("correlation", correlation_func);
    context.set_function("covariance", covariance_func);
    context.set_function("percentile", percentile_func);
}

}
