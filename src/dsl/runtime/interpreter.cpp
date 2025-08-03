#include "interpreter.h"
#include "engine/facade/facade.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <regex>

namespace dsl::runtime {

// Environment implementation
void Environment::define(const std::string& name, const Value& value) {
    variables_[name] = value;
}

Value Environment::get(const std::string& name) {
    if (variables_.find(name) != variables_.end()) {
        return variables_[name];
    }
    
    if (enclosing_ != nullptr) {
        return enclosing_->get(name);
    }
    
    throw std::runtime_error("Undefined variable '" + name + "'.");
}

void Environment::assign(const std::string& name, const Value& value) {
    if (variables_.find(name) != variables_.end()) {
        variables_[name] = value;
        return;
    }
    
    if (enclosing_ != nullptr) {
        enclosing_->assign(name, value);
        return;
    }
    
    throw std::runtime_error("Undefined variable '" + name + "'.");
}

// Interpreter implementation
Interpreter::Interpreter() : environment_(&globals_), program_(nullptr) {
    register_builtins();
}

void Interpreter::register_builtins() {
    // Get the singleton instance of the engine facade
    auto& engine = sep::engine::EngineFacade::getInstance();
    
    // AGI Engine Bridge Functions - THE REAL POWER
    builtins_["measure_coherence"] = [&engine](const std::vector<Value>& args) -> Value {
        std::cout << "DSL: Calling real measure_coherence with " << args.size() << " arguments" << std::endl;
        
        sep::engine::PatternAnalysisRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "default_pattern";
            }
        }
        request.analysis_depth = 3;
        request.include_relationships = true;
        
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            return static_cast<double>(response.confidence_score);
        } else {
            throw std::runtime_error("Engine call failed for measure_coherence");
        }
    };
    
    builtins_["qfh_analyze"] = [&engine](const std::vector<Value>& args) -> Value {
        if (args.empty()) {
            throw std::runtime_error("qfh_analyze expects a bitstream argument");
        }

        std::vector<uint8_t> bitstream;
        try {
            std::string bitstream_str = std::any_cast<std::string>(args[0]);
            for (char c : bitstream_str) {
                bitstream.push_back(c - '0');
            }
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid bitstream argument for qfh_analyze");
        }

        sep::engine::QFHAnalysisRequest request;
        request.bitstream = bitstream;
        sep::engine::QFHAnalysisResponse response;
        auto result = engine.qfhAnalyze(request, response);

        if (sep::core::isSuccess(result)) {
            return static_cast<double>(response.rupture_ratio);
        } else {
            throw std::runtime_error("Engine call failed for qfh_analyze");
        }
    };
    
    builtins_["measure_entropy"] = [&engine](const std::vector<Value>& args) -> Value {
        std::cout << "DSL: Calling real measure_entropy with " << args.size() << " arguments" << std::endl;
        
        sep::engine::PatternAnalysisRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "entropy_pattern";
            }
        }
        request.analysis_depth = 2;
        request.include_relationships = false;
        
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            std::cout << "Real entropy from engine: " << response.entropy << std::endl;
            return static_cast<double>(response.entropy);
        } else {
            throw std::runtime_error("Engine call failed for measure_entropy");
        }
    };
    
    builtins_["extract_bits"] = [&engine](const std::vector<Value>& args) -> Value {
        std::cout << "DSL: Calling real extract_bits with " << args.size() << " arguments" << std::endl;
        
        sep::engine::BitExtractionRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "bitstream_pattern";
            }
        }
        
        sep::engine::BitExtractionResponse response;
        auto result = engine.extractBits(request, response);
        
        if (sep::core::isSuccess(result) && response.success) {
            // Convert bitstream to string for DSL use
            std::string bitstream_str;
            for (uint8_t bit : response.bitstream) {
                bitstream_str += (bit ? '1' : '0');
            }
            return bitstream_str;
        } else {
            throw std::runtime_error("Engine call failed for extract_bits");
        }
    };
    
    builtins_["manifold_optimize"] = [&engine](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("manifold_optimize expects pattern_id, target_coherence, target_stability");
        }
        
        sep::engine::ManifoldOptimizationRequest request;
        try {
            request.pattern_id = std::any_cast<std::string>(args[0]);
            request.target_coherence = static_cast<float>(std::any_cast<double>(args[1]));
            request.target_stability = static_cast<float>(std::any_cast<double>(args[2]));
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid arguments for manifold_optimize");
        }
        
        sep::engine::ManifoldOptimizationResponse response;
        auto result = engine.manifoldOptimize(request, response);
        
        if (sep::core::isSuccess(result) && response.success) {
            return static_cast<double>(response.optimized_coherence);
        } else {
            throw std::runtime_error("Engine call failed for manifold_optimize");
        }
    };
    
    // Math functions
    builtins_["abs"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("abs() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::abs(value);
    };
    
    builtins_["sqrt"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("sqrt() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        if (value < 0) {
            throw std::runtime_error("sqrt() of negative number");
        }
        return std::sqrt(value);
    };
    
    builtins_["min"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("min() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::min(a, b);
    };
    
    builtins_["max"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("max() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::max(a, b);
    };
    
    // Trigonometric functions
    builtins_["sin"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("sin() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::sin(value);
    };
    
    builtins_["cos"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("cos() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::cos(value);
    };
    
    builtins_["tan"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("tan() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::tan(value);
    };
    
    builtins_["asin"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("asin() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        if (value < -1.0 || value > 1.0) {
            throw std::runtime_error("asin() argument out of domain [-1, 1]");
        }
        return std::asin(value);
    };
    
    builtins_["acos"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("acos() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        if (value < -1.0 || value > 1.0) {
            throw std::runtime_error("acos() argument out of domain [-1, 1]");
        }
        return std::acos(value);
    };
    
    builtins_["atan"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("atan() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::atan(value);
    };
    
    builtins_["atan2"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("atan2() expects exactly 2 arguments");
        }
        double y = std::any_cast<double>(args[0]);
        double x = std::any_cast<double>(args[1]);
        return std::atan2(y, x);
    };
    
    // Exponential and logarithmic functions
    builtins_["exp"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("exp() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::exp(value);
    };
    
    builtins_["log"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("log() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        if (value <= 0) {
            throw std::runtime_error("log() of non-positive number");
        }
        return std::log(value);
    };
    
    builtins_["log10"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("log10() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        if (value <= 0) {
            throw std::runtime_error("log10() of non-positive number");
        }
        return std::log10(value);
    };
    
    builtins_["log2"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("log2() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        if (value <= 0) {
            throw std::runtime_error("log2() of non-positive number");
        }
        return std::log2(value);
    };
    
    // Power functions
    builtins_["pow"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("pow() expects exactly 2 arguments");
        }
        double base = std::any_cast<double>(args[0]);
        double exponent = std::any_cast<double>(args[1]);
        return std::pow(base, exponent);
    };
    
    builtins_["cbrt"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("cbrt() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::cbrt(value);
    };
    
    // Rounding and modular functions
    builtins_["floor"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("floor() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::floor(value);
    };
    
    builtins_["ceil"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("ceil() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::ceil(value);
    };
    
    builtins_["round"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("round() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::round(value);
    };
    
    builtins_["trunc"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("trunc() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::trunc(value);
    };
    
    builtins_["fmod"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("fmod() expects exactly 2 arguments");
        }
        double x = std::any_cast<double>(args[0]);
        double y = std::any_cast<double>(args[1]);
        if (y == 0) {
            throw std::runtime_error("fmod() division by zero");
        }
        return std::fmod(x, y);
    };
    
    // Mathematical constants
    builtins_["pi"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 0) {
            throw std::runtime_error("pi() expects no arguments");
        }
        return M_PI;
    };
    
    builtins_["e"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 0) {
            throw std::runtime_error("e() expects no arguments");
        }
        return M_E;
    };
    
    // Statistical functions (for variable number of arguments)
    builtins_["mean"] = [](const std::vector<Value>& args) -> Value {
        if (args.empty()) {
            throw std::runtime_error("mean() expects at least 1 argument");
        }
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        return sum / args.size();
    };
    
    builtins_["sum"] = [](const std::vector<Value>& args) -> Value {
        if (args.empty()) {
            throw std::runtime_error("sum() expects at least 1 argument");
        }
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        return sum;
    };
    
    builtins_["count"] = [](const std::vector<Value>& args) -> Value {
        return static_cast<double>(args.size());
    };
    
    builtins_["stddev"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("stddev() expects at least 2 arguments");
        }
        
        // Calculate mean
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        double mean = sum / args.size();
        
        // Calculate variance
        double variance = 0.0;
        for (const auto& arg : args) {
            double val = std::any_cast<double>(arg);
            variance += (val - mean) * (val - mean);
        }
        variance /= (args.size() - 1); // Sample standard deviation
        
        return std::sqrt(variance);
    };
    
    builtins_["variance"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("variance() expects at least 2 arguments");
        }
        
        // Calculate mean
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        double mean = sum / args.size();
        
        // Calculate variance
        double variance = 0.0;
        for (const auto& arg : args) {
            double val = std::any_cast<double>(arg);
            variance += (val - mean) * (val - mean);
        }
        return variance / (args.size() - 1); // Sample variance
    };
    
    builtins_["median"] = [](const std::vector<Value>& args) -> Value {
        if (args.empty()) {
            throw std::runtime_error("median() expects at least 1 argument");
        }
        
        // Convert to vector and sort
        std::vector<double> values;
        for (const auto& arg : args) {
            values.push_back(std::any_cast<double>(arg));
        }
        std::sort(values.begin(), values.end());
        
        size_t n = values.size();
        if (n % 2 == 0) {
            // Even number of elements - average of middle two
            return (values[n/2 - 1] + values[n/2]) / 2.0;
        } else {
            // Odd number of elements - middle element
            return values[n/2];
        }
    };
    
    builtins_["percentile"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("percentile() expects at least 2 arguments (percentile, values...)");
        }
        
        double p = std::any_cast<double>(args[0]);
        if (p < 0.0 || p > 100.0) {
            throw std::runtime_error("percentile() percentile must be between 0 and 100");
        }
        
        // Convert remaining args to vector and sort
        std::vector<double> values;
        for (size_t i = 1; i < args.size(); i++) {
            values.push_back(std::any_cast<double>(args[i]));
        }
        std::sort(values.begin(), values.end());
        
        double index = (p / 100.0) * (values.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));
        
        if (lower == upper) {
            return values[lower];
        } else {
            double weight = index - lower;
            return values[lower] * (1.0 - weight) + values[upper] * weight;
        }
    };
    
    builtins_["correlation"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() % 2 != 0) {
            throw std::runtime_error("correlation() expects pairs of values (even number of arguments)");
        }
        if (args.size() < 4) {
            throw std::runtime_error("correlation() expects at least 2 pairs of values");
        }
        
        size_t n = args.size() / 2;
        std::vector<double> x, y;
        
        // Split into x and y arrays
        for (size_t i = 0; i < n; i++) {
            x.push_back(std::any_cast<double>(args[i]));
            y.push_back(std::any_cast<double>(args[i + n]));
        }
        
        // Calculate means
        double mean_x = 0.0, mean_y = 0.0;
        for (size_t i = 0; i < n; i++) {
            mean_x += x[i];
            mean_y += y[i];
        }
        mean_x /= n;
        mean_y /= n;
        
        // Calculate correlation coefficient
        double numerator = 0.0, sum_x_sq = 0.0, sum_y_sq = 0.0;
        for (size_t i = 0; i < n; i++) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_x_sq += dx * dx;
            sum_y_sq += dy * dy;
        }
        
        double denominator = std::sqrt(sum_x_sq * sum_y_sq);
        if (denominator == 0.0) {
            return 0.0; // No correlation if no variance
        }
        
        return numerator / denominator;
    };
    
    // Time series functions
    builtins_["moving_average"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("moving_average() expects at least 2 arguments (window_size, data...)");
        }
        
        int window_size = static_cast<int>(std::any_cast<double>(args[0]));
        if (window_size <= 0) {
            throw std::runtime_error("moving_average() window size must be positive");
        }
        
        // Convert remaining args to data array
        std::vector<double> data;
        for (size_t i = 1; i < args.size(); i++) {
            data.push_back(std::any_cast<double>(args[i]));
        }
        
        if (static_cast<int>(data.size()) < window_size) {
            throw std::runtime_error("moving_average() data size must be >= window size");
        }
        
        // Calculate moving averages
        std::vector<Value> result;
        for (size_t i = 0; i <= data.size() - window_size; i++) {
            double sum = 0.0;
            for (int j = 0; j < window_size; j++) {
                sum += data[i + j];
            }
            result.push_back(sum / window_size);
        }
        
        return result;
    };
    
    builtins_["exponential_moving_average"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("exponential_moving_average() expects at least 3 arguments (alpha, initial_value, data...)");
        }
        
        double alpha = std::any_cast<double>(args[0]);
        if (alpha <= 0.0 || alpha > 1.0) {
            throw std::runtime_error("exponential_moving_average() alpha must be between 0 and 1");
        }
        
        double ema = std::any_cast<double>(args[1]); // Initial value
        std::vector<Value> result;
        result.push_back(ema);
        
        // Apply EMA formula: EMA = alpha * current + (1 - alpha) * previous_EMA
        for (size_t i = 2; i < args.size(); i++) {
            double current = std::any_cast<double>(args[i]);
            ema = alpha * current + (1.0 - alpha) * ema;
            result.push_back(ema);
        }
        
        return result;
    };
    
    builtins_["trend_detection"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("trend_detection() expects at least 3 arguments (threshold, data...)");
        }
        
        double threshold = std::any_cast<double>(args[0]);
        
        // Convert args to data array
        std::vector<double> data;
        for (size_t i = 1; i < args.size(); i++) {
            data.push_back(std::any_cast<double>(args[i]));
        }
        
        if (data.size() < 2) {
            return 0.0; // No trend with less than 2 points
        }
        
        // Simple linear trend detection using least squares
        size_t n = data.size();
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
        
        for (size_t i = 0; i < n; i++) {
            double x = static_cast<double>(i);
            double y = data[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        // Calculate slope (trend strength)
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        // Return trend direction: 1 = upward, -1 = downward, 0 = no trend
        if (std::abs(slope) < threshold) {
            return 0.0; // No significant trend
        } else if (slope > 0) {
            return 1.0; // Upward trend
        } else {
            return -1.0; // Downward trend
        }
    };
    
    builtins_["rate_of_change"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("rate_of_change() expects at least 2 arguments (data...)");
        }
        
        // Convert args to data array
        std::vector<double> data;
        for (const auto& arg : args) {
            data.push_back(std::any_cast<double>(arg));
        }
        
        // Calculate rate of change between consecutive points
        std::vector<Value> result;
        for (size_t i = 1; i < data.size(); i++) {
            double change = data[i] - data[i-1];
            double rate = (data[i-1] != 0.0) ? (change / data[i-1]) * 100.0 : 0.0;
            result.push_back(rate);
        }
        
        return result;
    };
    
    // Data transformation functions
    builtins_["normalize"] = [](const std::vector<Value>& args) -> Value {
        if (args.empty()) {
            throw std::runtime_error("normalize() expects at least 1 argument (data...)");
        }
        
        // Convert args to data array
        std::vector<double> data;
        for (const auto& arg : args) {
            data.push_back(std::any_cast<double>(arg));
        }
        
        if (data.size() == 1) {
            return std::vector<Value>{1.0}; // Single value normalizes to 1
        }
        
        // Find min and max
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        
        if (min_val == max_val) {
            // All values are the same, return array of ones
            std::vector<Value> result(data.size(), 1.0);
            return result;
        }
        
        // Normalize to [0, 1] range
        std::vector<Value> result;
        for (double val : data) {
            double normalized = (val - min_val) / (max_val - min_val);
            result.push_back(normalized);
        }
        
        return result;
    };
    
    builtins_["standardize"] = [](const std::vector<Value>& args) -> Value {
        if (args.empty()) {
            throw std::runtime_error("standardize() expects at least 1 argument (data...)");
        }
        
        // Convert args to data array
        std::vector<double> data;
        for (const auto& arg : args) {
            data.push_back(std::any_cast<double>(arg));
        }
        
        if (data.size() == 1) {
            return std::vector<Value>{0.0}; // Single value standardizes to 0
        }
        
        // Calculate mean
        double sum = 0.0;
        for (double val : data) {
            sum += val;
        }
        double mean = sum / data.size();
        
        // Calculate standard deviation
        double variance_sum = 0.0;
        for (double val : data) {
            variance_sum += (val - mean) * (val - mean);
        }
        double std_dev = std::sqrt(variance_sum / (data.size() - 1));
        
        if (std_dev == 0.0) {
            // All values are the same, return array of zeros
            std::vector<Value> result(data.size(), 0.0);
            return result;
        }
        
        // Standardize: (value - mean) / std_dev
        std::vector<Value> result;
        for (double val : data) {
            double standardized = (val - mean) / std_dev;
            result.push_back(standardized);
        }
        
        return result;
    };
    
    builtins_["scale"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("scale() expects at least 3 arguments (min_target, max_target, data...)");
        }
        
        double min_target = std::any_cast<double>(args[0]);
        double max_target = std::any_cast<double>(args[1]);
        
        if (min_target >= max_target) {
            throw std::runtime_error("scale() min_target must be less than max_target");
        }
        
        // Convert remaining args to data array
        std::vector<double> data;
        for (size_t i = 2; i < args.size(); i++) {
            data.push_back(std::any_cast<double>(args[i]));
        }
        
        if (data.size() == 1) {
            return std::vector<Value>{(min_target + max_target) / 2.0}; // Single value goes to midpoint
        }
        
        // Find min and max of data
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        
        if (min_val == max_val) {
            // All values are the same, scale to midpoint
            double midpoint = (min_target + max_target) / 2.0;
            std::vector<Value> result(data.size(), midpoint);
            return result;
        }
        
        // Scale to [min_target, max_target] range
        std::vector<Value> result;
        for (double val : data) {
            double scaled = min_target + (val - min_val) * (max_target - min_target) / (max_val - min_val);
            result.push_back(scaled);
        }
        
        return result;
    };
    
    builtins_["filter_above"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("filter_above() expects at least 2 arguments (threshold, data...)");
        }
        
        double threshold = std::any_cast<double>(args[0]);
        
        // Filter values above threshold
        std::vector<Value> result;
        for (size_t i = 1; i < args.size(); i++) {
            double val = std::any_cast<double>(args[i]);
            if (val > threshold) {
                result.push_back(val);
            }
        }
        
        return result;
    };
    
    builtins_["filter_below"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("filter_below() expects at least 2 arguments (threshold, data...)");
        }
        
        double threshold = std::any_cast<double>(args[0]);
        
        // Filter values below threshold
        std::vector<Value> result;
        for (size_t i = 1; i < args.size(); i++) {
            double val = std::any_cast<double>(args[i]);
            if (val < threshold) {
                result.push_back(val);
            }
        }
        
        return result;
    };
    
    builtins_["filter_range"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("filter_range() expects at least 3 arguments (min_val, max_val, data...)");
        }
        
        double min_val = std::any_cast<double>(args[0]);
        double max_val = std::any_cast<double>(args[1]);
        
        if (min_val >= max_val) {
            throw std::runtime_error("filter_range() min_val must be less than max_val");
        }
        
        // Filter values within range [min_val, max_val]
        std::vector<Value> result;
        for (size_t i = 2; i < args.size(); i++) {
            double val = std::any_cast<double>(args[i]);
            if (val >= min_val && val <= max_val) {
                result.push_back(val);
            }
        }
        
        return result;
    };
    
    builtins_["clamp"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("clamp() expects at least 3 arguments (min_val, max_val, data...)");
        }
        
        double min_val = std::any_cast<double>(args[0]);
        double max_val = std::any_cast<double>(args[1]);
        
        if (min_val >= max_val) {
            throw std::runtime_error("clamp() min_val must be less than max_val");
        }
        
        // Clamp values to range [min_val, max_val]
        std::vector<Value> result;
        for (size_t i = 2; i < args.size(); i++) {
            double val = std::any_cast<double>(args[i]);
            double clamped = std::max(min_val, std::min(max_val, val));
            result.push_back(clamped);
        }
        
        return result;
    };
    
    // Pattern matching functions
    builtins_["regex_match"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("regex_match() expects 2 arguments (pattern, text)");
        }
        
        std::string pattern = std::any_cast<std::string>(args[0]);
        std::string text = std::any_cast<std::string>(args[1]);
        
        try {
            std::regex regex_pattern(pattern);
            return std::regex_search(text, regex_pattern);
        } catch (const std::regex_error& e) {
            throw std::runtime_error("regex_match() invalid regex pattern: " + std::string(e.what()));
        }
    };
    
    builtins_["regex_extract"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("regex_extract() expects 2 arguments (pattern, text)");
        }
        
        std::string pattern = std::any_cast<std::string>(args[0]);
        std::string text = std::any_cast<std::string>(args[1]);
        
        try {
            std::regex regex_pattern(pattern);
            std::smatch matches;
            
            if (std::regex_search(text, matches, regex_pattern)) {
                std::vector<Value> result;
                for (const auto& match : matches) {
                    result.push_back(match.str());
                }
                return result;
            } else {
                return std::vector<Value>();
            }
        } catch (const std::regex_error& e) {
            throw std::runtime_error("regex_extract() invalid regex pattern: " + std::string(e.what()));
        }
    };
    
    builtins_["regex_replace"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 3) {
            throw std::runtime_error("regex_replace() expects 3 arguments (pattern, replacement, text)");
        }
        
        std::string pattern = std::any_cast<std::string>(args[0]);
        std::string replacement = std::any_cast<std::string>(args[1]);
        std::string text = std::any_cast<std::string>(args[2]);
        
        try {
            std::regex regex_pattern(pattern);
            return std::regex_replace(text, regex_pattern, replacement);
        } catch (const std::regex_error& e) {
            throw std::runtime_error("regex_replace() invalid regex pattern: " + std::string(e.what()));
        }
    };
    
    builtins_["fuzzy_match"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2 && args.size() != 3) {
            throw std::runtime_error("fuzzy_match() expects 2 or 3 arguments (text1, text2, [threshold])");
        }
        
        std::string text1 = std::any_cast<std::string>(args[0]);
        std::string text2 = std::any_cast<std::string>(args[1]);
        double threshold = (args.size() == 3) ? std::any_cast<double>(args[2]) : 0.8;
        
        // Implement Levenshtein distance-based fuzzy matching
        auto levenshtein_distance = [](const std::string& s1, const std::string& s2) -> int {
            size_t len1 = s1.length();
            size_t len2 = s2.length();
            
            std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));
            
            for (size_t i = 0; i <= len1; ++i) dp[i][0] = i;
            for (size_t j = 0; j <= len2; ++j) dp[0][j] = j;
            
            for (size_t i = 1; i <= len1; ++i) {
                for (size_t j = 1; j <= len2; ++j) {
                    if (s1[i-1] == s2[j-1]) {
                        dp[i][j] = dp[i-1][j-1];
                    } else {
                        dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
                    }
                }
            }
            
            return dp[len1][len2];
        };
        
        int distance = levenshtein_distance(text1, text2);
        int max_len = std::max(text1.length(), text2.length());
        
        if (max_len == 0) return true; // Both strings are empty
        
        double similarity = 1.0 - (double)distance / max_len;
        return similarity >= threshold;
    };
    
    builtins_["fuzzy_similarity"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("fuzzy_similarity() expects 2 arguments (text1, text2)");
        }
        
        std::string text1 = std::any_cast<std::string>(args[0]);
        std::string text2 = std::any_cast<std::string>(args[1]);
        
        // Implement Levenshtein distance-based similarity
        auto levenshtein_distance = [](const std::string& s1, const std::string& s2) -> int {
            size_t len1 = s1.length();
            size_t len2 = s2.length();
            
            std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));
            
            for (size_t i = 0; i <= len1; ++i) dp[i][0] = i;
            for (size_t j = 0; j <= len2; ++j) dp[0][j] = j;
            
            for (size_t i = 1; i <= len1; ++i) {
                for (size_t j = 1; j <= len2; ++j) {
                    if (s1[i-1] == s2[j-1]) {
                        dp[i][j] = dp[i-1][j-1];
                    } else {
                        dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
                    }
                }
            }
            
            return dp[len1][len2];
        };
        
        int distance = levenshtein_distance(text1, text2);
        int max_len = std::max(text1.length(), text2.length());
        
        if (max_len == 0) return 1.0; // Both strings are empty, perfect match
        
        return 1.0 - (double)distance / max_len;
    };
    
    // Aggregation functions
    builtins_["groupby"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("groupby() expects at least 2 arguments (key_func, data...)");
        }
        
        // For now, implement a simple groupby based on value equality
        // In a more advanced implementation, we'd support function-based grouping
        
        std::map<std::string, std::vector<Value>> groups;
        
        for (size_t i = 1; i < args.size(); i++) {
            std::string key = std::any_cast<std::string>(args[0]) + "_" + std::to_string(i);
            
            // Try to extract a grouping key from the value
            std::string group_key;
            try {
                if (args[i].type() == typeid(double)) {
                    double val = std::any_cast<double>(args[i]);
                    group_key = (val < 0) ? "negative" : (val == 0) ? "zero" : "positive";
                } else if (args[i].type() == typeid(std::string)) {
                    std::string str = std::any_cast<std::string>(args[i]);
                    group_key = str.empty() ? "empty" : str.substr(0, 1); // Group by first character
                } else {
                    group_key = "other";
                }
            } catch (...) {
                group_key = "unknown";
            }
            
            groups[group_key].push_back(args[i]);
        }
        
        // Convert to array of arrays format
        std::vector<Value> result;
        for (const auto& [key, values] : groups) {
            std::vector<Value> group_data;
            group_data.push_back(key); // Group key as first element
            group_data.insert(group_data.end(), values.begin(), values.end());
            result.push_back(group_data);
        }
        
        return result;
    };
    
    builtins_["pivot"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("pivot() expects at least 3 arguments (rows, cols, values)");
        }
        
        // Simple pivot implementation - transpose data based on row/column indices
        try {
            std::vector<Value> rows = std::any_cast<std::vector<Value>>(args[0]);
            std::vector<Value> cols = std::any_cast<std::vector<Value>>(args[1]);
            std::vector<Value> values = std::any_cast<std::vector<Value>>(args[2]);
            
            if (rows.size() != cols.size() || rows.size() != values.size()) {
                throw std::runtime_error("pivot() arrays must have the same length");
            }
            
            // Create pivot table as nested arrays
            std::map<std::string, std::map<std::string, double>> pivot_table;
            
            for (size_t i = 0; i < rows.size(); i++) {
                std::string row_key = std::any_cast<std::string>(rows[i]);
                std::string col_key = std::any_cast<std::string>(cols[i]);
                double value = std::any_cast<double>(values[i]);
                
                pivot_table[row_key][col_key] = value;
            }
            
            // Convert to nested array format
            std::vector<Value> result;
            for (const auto& [row_key, row_data] : pivot_table) {
                std::vector<Value> row;
                row.push_back(row_key);
                for (const auto& [col_key, value] : row_data) {
                    row.push_back(value);
                }
                result.push_back(row);
            }
            
            return result;
            
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("pivot() arguments must be arrays");
        }
    };
    
    builtins_["rollup"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 2) {
            throw std::runtime_error("rollup() expects at least 2 arguments (aggregation_func, data...)");
        }
        
        std::string agg_func = std::any_cast<std::string>(args[0]);
        
        // Extract all numeric data points
        std::vector<double> data;
        for (size_t i = 1; i < args.size(); i++) {
            try {
                if (args[i].type() == typeid(std::vector<Value>)) {
                    // Handle array argument
                    std::vector<Value> arr = std::any_cast<std::vector<Value>>(args[i]);
                    for (const auto& val : arr) {
                        data.push_back(std::any_cast<double>(val));
                    }
                } else {
                    // Handle individual value
                    data.push_back(std::any_cast<double>(args[i]));
                }
            } catch (const std::bad_any_cast& e) {
                // Skip non-numeric values
                continue;
            }
        }
        
        if (data.empty()) {
            throw std::runtime_error("rollup() no numeric data found");
        }
        
        // Apply aggregation function
        if (agg_func == "sum") {
            double sum = 0.0;
            for (double val : data) {
                sum += val;
            }
            return sum;
        } else if (agg_func == "avg" || agg_func == "mean") {
            double sum = 0.0;
            for (double val : data) {
                sum += val;
            }
            return sum / data.size();
        } else if (agg_func == "min") {
            return *std::min_element(data.begin(), data.end());
        } else if (agg_func == "max") {
            return *std::max_element(data.begin(), data.end());
        } else if (agg_func == "count") {
            return (double)data.size();
        } else {
            throw std::runtime_error("rollup() unsupported aggregation function: " + agg_func);
        }
    };
    
    builtins_["aggregate"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("aggregate() expects at least 3 arguments (data, group_key, agg_func)");
        }
        
        try {
            std::vector<Value> data = std::any_cast<std::vector<Value>>(args[0]);
            std::string group_key = std::any_cast<std::string>(args[1]);
            std::string agg_func = std::any_cast<std::string>(args[2]);
            
            // For simplicity, group by value ranges for numeric data
            std::map<std::string, std::vector<double>> groups;
            
            for (const auto& item : data) {
                if (item.type() == typeid(double)) {
                    double val = std::any_cast<double>(item);
                    std::string key;
                    
                    if (group_key == "range") {
                        if (val < 10) key = "0-10";
                        else if (val < 50) key = "10-50";
                        else if (val < 100) key = "50-100";
                        else key = "100+";
                    } else if (group_key == "sign") {
                        key = (val < 0) ? "negative" : (val == 0) ? "zero" : "positive";
                    } else {
                        key = "all";
                    }
                    
                    groups[key].push_back(val);
                }
            }
            
            // Apply aggregation to each group
            std::vector<Value> result;
            for (const auto& [key, group_data] : groups) {
                std::vector<Value> group_result;
                group_result.push_back(key);
                
                double agg_value = 0.0;
                if (agg_func == "sum") {
                    for (double val : group_data) agg_value += val;
                } else if (agg_func == "avg" || agg_func == "mean") {
                    for (double val : group_data) agg_value += val;
                    agg_value /= group_data.size();
                } else if (agg_func == "count") {
                    agg_value = (double)group_data.size();
                } else if (agg_func == "min") {
                    agg_value = *std::min_element(group_data.begin(), group_data.end());
                } else if (agg_func == "max") {
                    agg_value = *std::max_element(group_data.begin(), group_data.end());
                }
                
                group_result.push_back(agg_value);
                result.push_back(group_result);
            }
            
            return result;
            
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("aggregate() invalid argument types");
        }
    };
    
    // Debugging functions
    builtins_["print"] = [this](const std::vector<Value>& args) -> Value {
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << stringify(args[i]);
        }
        std::cout << std::endl;
        return 0.0; // Return dummy value
    };
}

void Interpreter::interpret(const ast::Program& program) {
    environment_ = &globals_;
    program_ = &program;
    
    try {
        // Execute stream declarations
        for (const auto& stream : program.streams) {
            execute_stream_decl(*stream);
        }
        
        // Execute pattern declarations
        for (const auto& pattern : program.patterns) {
            execute_pattern_decl(*pattern);
        }
        
        // Execute signal declarations
        for (const auto& signal : program.signals) {
            execute_signal_decl(*signal);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
    }
}

Value Interpreter::evaluate(const ast::Expression& expr) {
    // Use dynamic casting to determine the actual type
    if (const auto* number = dynamic_cast<const ast::NumberLiteral*>(&expr)) {
        return visit_number_literal(*number);
    }
    if (const auto* string_lit = dynamic_cast<const ast::StringLiteral*>(&expr)) {
        return visit_string_literal(*string_lit);
    }
    if (const auto* boolean_lit = dynamic_cast<const ast::BooleanLiteral*>(&expr)) {
        return visit_boolean_literal(*boolean_lit);
    }
    if (const auto* identifier = dynamic_cast<const ast::Identifier*>(&expr)) {
        return visit_identifier(*identifier);
    }
    if (const auto* binary_op = dynamic_cast<const ast::BinaryOp*>(&expr)) {
        return visit_binary_op(*binary_op);
    }
    if (const auto* unary_op = dynamic_cast<const ast::UnaryOp*>(&expr)) {
        return visit_unary_op(*unary_op);
    }
    if (const auto* call = dynamic_cast<const ast::Call*>(&expr)) {
        return visit_call(*call);
    }
    if (const auto* member_access = dynamic_cast<const ast::MemberAccess*>(&expr)) {
        return visit_member_access(*member_access);
    }
    if (const auto* weighted_sum = dynamic_cast<const ast::WeightedSum*>(&expr)) {
       return visit_weighted_sum(*weighted_sum);
    }
    if (const auto* await_expr = dynamic_cast<const ast::AwaitExpression*>(&expr)) {
        return visit_await_expression(*await_expr);
    }
    if (const auto* array_literal = dynamic_cast<const ast::ArrayLiteral*>(&expr)) {
        return visit_array_literal(*array_literal);
    }
    if (const auto* array_access = dynamic_cast<const ast::ArrayAccess*>(&expr)) {
        return visit_array_access(*array_access);
    }
    
    throw std::runtime_error("Unknown expression type");
}

void Interpreter::execute(const ast::Statement& stmt) {
    if (const auto* assignment = dynamic_cast<const ast::Assignment*>(&stmt)) {
        visit_assignment(*assignment);
    } else if (const auto* expr_stmt = dynamic_cast<const ast::ExpressionStatement*>(&stmt)) {
        visit_expression_statement(*expr_stmt);
    } else if (const auto* evolve_stmt = dynamic_cast<const ast::EvolveStatement*>(&stmt)) {
        visit_evolve_statement(*evolve_stmt);
    } else if (const auto* if_stmt = dynamic_cast<const ast::IfStatement*>(&stmt)) {
        visit_if_statement(*if_stmt);
    } else if (const auto* while_stmt = dynamic_cast<const ast::WhileStatement*>(&stmt)) {
        visit_while_statement(*while_stmt);
    } else if (const auto* func_decl = dynamic_cast<const ast::FunctionDeclaration*>(&stmt)) {
        visit_function_declaration(*func_decl);
    } else if (const auto* return_stmt = dynamic_cast<const ast::ReturnStatement*>(&stmt)) {
        visit_return_statement(*return_stmt);
    } else if (const auto* import_stmt = dynamic_cast<const ast::ImportStatement*>(&stmt)) {
        visit_import_statement(*import_stmt);
    } else if (const auto* export_stmt = dynamic_cast<const ast::ExportStatement*>(&stmt)) {
        visit_export_statement(*export_stmt);
    } else if (const auto* async_func_decl = dynamic_cast<const ast::AsyncFunctionDeclaration*>(&stmt)) {
        visit_async_function_declaration(*async_func_decl);
    } else if (const auto* try_stmt = dynamic_cast<const ast::TryStatement*>(&stmt)) {
        visit_try_statement(*try_stmt);
    } else if (const auto* throw_stmt = dynamic_cast<const ast::ThrowStatement*>(&stmt)) {
        visit_throw_statement(*throw_stmt);
    } else {
        throw std::runtime_error("Unknown statement type");
    }
}

Value Interpreter::visit_number_literal(const ast::NumberLiteral& node) {
    return node.value;
}

Value Interpreter::visit_string_literal(const ast::StringLiteral& node) {
    return node.value;
}

Value Interpreter::visit_boolean_literal(const ast::BooleanLiteral& node) {
    return node.value;
}

Value Interpreter::visit_identifier(const ast::Identifier& node) {
    return environment_->get(node.name);
}

Value Interpreter::visit_binary_op(const ast::BinaryOp& node) {
    Value left = evaluate(*node.left);
    Value right = evaluate(*node.right);
    
    if (node.op == "+") {
        // Handle both numeric and string concatenation
        try {
            double left_num = std::any_cast<double>(left);
            double right_num = std::any_cast<double>(right);
            return left_num + right_num;
        } catch (const std::bad_any_cast&) {
            std::string left_str = stringify(left);
            std::string right_str = stringify(right);
            return left_str + right_str;
        }
    }
    if (node.op == "-") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num - right_num;
    }
    if (node.op == "*") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num * right_num;
    }
    if (node.op == "/") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        if (right_num == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        return left_num / right_num;
    }
    if (node.op == ">") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num > right_num;
    }
    if (node.op == "<") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num < right_num;
    }
    if (node.op == ">=") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num >= right_num;
    }
    if (node.op == "<=") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num <= right_num;
    }
    if (node.op == "==") {
        return is_equal(left, right);
    }
    if (node.op == "!=") {
        return !is_equal(left, right);
    }
    if (node.op == "&&") {
        bool left_bool = std::any_cast<bool>(left);
        bool right_bool = std::any_cast<bool>(right);
        return left_bool && right_bool;
    }
    if (node.op == "||") {
        bool left_bool = std::any_cast<bool>(left);
        bool right_bool = std::any_cast<bool>(right);
        return left_bool || right_bool;
    }
    
    throw std::runtime_error("Unknown binary operator: " + node.op);
}

Value Interpreter::visit_unary_op(const ast::UnaryOp& node) {
    Value right = evaluate(*node.right);
    
    if (node.op == "-") {
        double right_num = std::any_cast<double>(right);
        return -right_num;
    }
    if (node.op == "!") {
        bool right_bool = std::any_cast<bool>(right);
        return !right_bool;
    }
    
    throw std::runtime_error("Unknown unary operator: " + node.op);
}

Value Interpreter::visit_call(const ast::Call& node) {
    std::vector<Value> arguments;
    for (const auto& arg : node.args) {
        arguments.push_back(evaluate(*arg));
    }

    // First check if it's a user-defined function
    try {
        Value callee = environment_->get(node.callee);
        if (auto function = std::any_cast<std::shared_ptr<UserFunction>>(callee)) {
            return function->call(*this, arguments);
        }
    } catch (const std::runtime_error& e) {
        // Not a user-defined function, try builtin
    } catch (const std::bad_any_cast&) {
        // Not a user-defined function, try builtin
    }
    
    // Fall back to builtin function
    return call_builtin_function(node.callee, arguments);
}

Value Interpreter::visit_member_access(const ast::MemberAccess& node) {
    Value object = evaluate(*node.object);
    std::cout << "Member access: looking for member '" << node.member << "'" << std::endl;
    
    try {
        // Try to cast object to PatternResult
        PatternResult pattern_result = std::any_cast<PatternResult>(object);
        std::cout << "Found pattern result with " << pattern_result.size() << " members" << std::endl;
        
        // Debug: print all available members
        for (const auto& [name, value] : pattern_result) {
            std::cout << "  Available member: " << name << std::endl;
        }
        
        // Look up the member in the pattern result
        auto it = pattern_result.find(node.member);
        if (it != pattern_result.end()) {
            std::cout << "Found member '" << node.member << "'" << std::endl;
            return it->second;
        } else {
            throw std::runtime_error("Pattern member not found: " + node.member);
        }
    } catch (const std::bad_any_cast&) {
        throw std::runtime_error("Cannot access member of non-pattern object");
    }
}

void Interpreter::visit_assignment(const ast::Assignment& node) {
    Value value = evaluate(*node.value);
    environment_->define(node.name, value);
}

void Interpreter::visit_expression_statement(const ast::ExpressionStatement& node) {
    Value result = evaluate(*node.expression);
    // For expression statements, we might want to print the result
    std::cout << stringify(result) << std::endl;
}

void Interpreter::execute_stream_decl(const ast::StreamDecl& decl) {
    std::cout << "Executing stream declaration: " << decl.name << " from " << decl.source << std::endl;
    
    // For now, just store the stream info in the environment
    environment_->define(decl.name + "_source", decl.source);
    
    // Store parameters
    for (const auto& param : decl.params) {
        environment_->define(decl.name + "_" + param.first, param.second);
    }
}

void Interpreter::execute_pattern_decl(const ast::PatternDecl& decl) {
    std::cout << "Executing pattern declaration: " << decl.name << std::endl;
    
    // Create a new environment for the pattern
    Environment pattern_env(environment_);
    Environment* previous = environment_;
    environment_ = &pattern_env;
    
    // Define inputs in the pattern environment
    for (const auto& input : decl.inputs) {
        environment_->define(input, 0.0);
    }
    
    // If this pattern inherits from another pattern, find and execute it first
    if (!decl.parent_pattern.empty()) {
        // Find the parent pattern in the program
        bool found_parent = false;
        for (const auto& pattern : program_->patterns) {
            if (pattern->name == decl.parent_pattern) {
                // Execute parent pattern in its own environment
                Environment parent_env(&globals_);  // Parent pattern uses globals as enclosing scope
                Environment* previous = environment_;
                environment_ = &parent_env;
                
                // Execute parent pattern
                execute_pattern_decl(*pattern);
                
                // Restore environment
                environment_ = previous;
                
                try {
                    // Get parent pattern result from globals
                    Value parent_result = globals_.get(decl.parent_pattern);
                    PatternResult parent_pattern = std::any_cast<PatternResult>(parent_result);
                    
                    // Copy parent pattern variables into current environment
                    for (const auto& [name, value] : parent_pattern) {
                        environment_->define(name, value);
                    }
                } catch (const std::exception& e) {
                    throw std::runtime_error("Failed to inherit from pattern '" + decl.parent_pattern + "': " + e.what());
                }
                
                found_parent = true;
                break;
            }
        }
        
        if (!found_parent) {
            throw std::runtime_error("Parent pattern '" + decl.parent_pattern + "' not found");
        }
    }
    
    // Execute pattern body
    for (const auto& stmt : decl.body) {
        execute(*stmt);
    }
    
    // Capture pattern results from the pattern environment
    PatternResult pattern_result;
    // Copy all variables from pattern_env to pattern_result
    // Note: This is a simplified approach - in practice you'd want to be more selective
    try {
        // Get all variables defined in this pattern scope
        const auto& pattern_vars = pattern_env.getVariables();
        for (const auto& [name, value] : pattern_vars) {
            pattern_result[name] = value;
        }
        
        // Store the pattern result in global environment under the pattern name
        globals_.define(decl.name, pattern_result);
        std::cout << "Pattern " << decl.name << " captured " << pattern_result.size() << " variables" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Warning: Could not capture pattern results: " << e.what() << std::endl;
    }
    
    // Restore previous environment
    environment_ = previous;
}

void Interpreter::execute_signal_decl(const ast::SignalDecl& decl) {
    std::cout << "Executing signal declaration: " << decl.name << std::endl;
    
    if (decl.trigger) {
        Value trigger_result = evaluate(*decl.trigger);
        bool should_trigger = is_truthy(trigger_result);
        
        if (should_trigger) {
            std::cout << "Signal " << decl.name << " triggered! Action: " << decl.action << std::endl;
            
            if (decl.confidence) {
                Value confidence_value = evaluate(*decl.confidence);
                std::cout << "Confidence: " << stringify(confidence_value) << std::endl;
            }
        }
    }
}

Value Interpreter::call_builtin_function(const std::string& name, const std::vector<Value>& args) {
    // First check the dynamic built-ins map
    auto it = builtins_.find(name);
    if (it != builtins_.end()) {
        return it->second(args);
    }
    
    // Fall back to legacy hardcoded functions (TODO: migrate all to builtins_ map)
    // Get the singleton instance of the engine facade
    auto& engine = sep::engine::EngineFacade::getInstance();
    
    if (name == "measure_coherence") {
        std::cout << "Calling real measure_coherence with " << args.size() << " arguments" << std::endl;
        
        // Convert DSL arguments into the request struct
        sep::engine::PatternAnalysisRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "default_pattern";
            }
        }
        request.analysis_depth = 3;
        request.include_relationships = true;
        
        // Call the real C++ function
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            return static_cast<double>(response.confidence_score);
        } else {
            throw std::runtime_error("Engine call failed for measure_coherence");
        }
    }
    
    if (name == "qfh_analyze") {
        if (args.empty()) {
            throw std::runtime_error("qfh_analyze expects a bitstream argument");
        }

        // Convert the DSL bitstream argument to a std::vector<uint8_t>
        std::vector<uint8_t> bitstream;
        try {
            std::string bitstream_str = std::any_cast<std::string>(args[0]);
            for (char c : bitstream_str) {
                bitstream.push_back(c - '0');
            }
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid bitstream argument for qfh_analyze");
        }

        // Call the engine facade
        sep::engine::QFHAnalysisRequest request;
        request.bitstream = bitstream;
        sep::engine::QFHAnalysisResponse response;
        auto result = engine.qfhAnalyze(request, response);

        if (sep::core::isSuccess(result)) {
            // For now, we'll return the rupture ratio as the primary result
            return static_cast<double>(response.rupture_ratio);
        } else {
            throw std::runtime_error("Engine call failed for qfh_analyze");
        }
    }
    
    if (name == "measure_entropy") {
        std::cout << "Calling real measure_entropy with " << args.size() << " arguments" << std::endl;
        
        // Convert DSL arguments into the request struct
        sep::engine::PatternAnalysisRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "entropy_pattern";
            }
        }
        request.analysis_depth = 2;
        request.include_relationships = false;
        
        // Call the real C++ function to get pattern metrics
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            // Return real entropy from QFH analysis
            std::cout << "Real entropy from engine: " << response.entropy << std::endl;
            return static_cast<double>(response.entropy);
        } else {
            throw std::runtime_error("Engine call failed for measure_entropy");
        }
    }
    
    if (name == "extract_bits") {
        std::cout << "Calling real extract_bits with " << args.size() << " arguments" << std::endl;
        
        // Convert DSL arguments into the bit extraction request
        sep::engine::BitExtractionRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "bitstream_pattern";
            }
        }
        
        // Call the real bit extraction engine
        sep::engine::BitExtractionResponse response;
        auto result = engine.extractBits(request, response);
        
        if (sep::core::isSuccess(result) && response.success) {
            // Convert bitstream to string representation for DSL
            std::string bitstream;
            for (uint8_t bit : response.bitstream) {
                bitstream += (bit == 1) ? '1' : '0';
            }
            
            std::cout << "Real bit extraction - extracted " << response.bitstream.size() << " bits" << std::endl;
            return bitstream;
        } else {
            throw std::runtime_error("Engine call failed for extract_bits: " + response.error_message);
        }
    }

    if (name == "manifold_optimize") {
        if (args.empty()) {
            throw std::runtime_error("manifold_optimize expects a pattern_id argument");
        }

        // Get the pattern_id from the DSL arguments
        std::string pattern_id;
        try {
            pattern_id = std::any_cast<std::string>(args[0]);
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid pattern_id argument for manifold_optimize");
        }

        // Call the engine facade
        sep::engine::ManifoldOptimizationRequest request;
        request.pattern_id = pattern_id;
        sep::engine::ManifoldOptimizationResponse response;
        auto result = engine.manifoldOptimize(request, response);

        if (sep::core::isSuccess(result)) {
            // For now, we'll return a boolean indicating success
            return response.success;
        } else {
            throw std::runtime_error("Engine call failed for manifold_optimize");
        }
    }
    
    // ============================================================================
    // Type Checking & Conversion Functions (TASK.md Phase 2A Priority 1)
    // ============================================================================
    if (name == "is_number") {
        if (args.empty()) {
            throw std::runtime_error("is_number() requires exactly 1 argument");
        }
        try {
            std::any_cast<double>(args[0]);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    
    if (name == "is_string") {
        if (args.empty()) {
            throw std::runtime_error("is_string() requires exactly 1 argument");
        }
        try {
            std::any_cast<std::string>(args[0]);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    
    if (name == "is_bool") {
        if (args.empty()) {
            throw std::runtime_error("is_bool() requires exactly 1 argument");
        }
        try {
            std::any_cast<bool>(args[0]);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    
    if (name == "to_string") {
        if (args.empty()) {
            throw std::runtime_error("to_string() requires exactly 1 argument");
        }
        return stringify(args[0]);
    }
    
    if (name == "to_number") {
        if (args.empty()) {
            throw std::runtime_error("to_number() requires exactly 1 argument");
        }
        try {
            return std::any_cast<double>(args[0]);
        } catch (const std::bad_any_cast&) {
            try {
                std::string str = std::any_cast<std::string>(args[0]);
                return std::stod(str);
            } catch (const std::exception&) {
                try {
                    bool b = std::any_cast<bool>(args[0]);
                    return b ? 1.0 : 0.0;
                } catch (const std::bad_any_cast&) {
                    throw std::runtime_error("Cannot convert this type to number");
                }
            }
        }
    }
    
    // ============================================================================
    // Math Functions
    // ============================================================================
    if (name == "abs") {
        if (args.size() != 1) {
            throw std::runtime_error("abs() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::abs(x);
    }
    
    if (name == "min") {
        if (args.size() != 2) {
            throw std::runtime_error("min() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::min(a, b);
    }
    
    if (name == "max") {
        if (args.size() != 2) {
            throw std::runtime_error("max() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::max(a, b);
    }
    
    if (name == "sqrt") {
        if (args.size() != 1) {
            throw std::runtime_error("sqrt() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        if (x < 0.0) {
            throw std::runtime_error("sqrt() domain error: argument must be non-negative");
        }
        return std::sqrt(x);
    }
    
    if (name == "pow") {
        if (args.size() != 2) {
            throw std::runtime_error("pow() expects exactly 2 arguments");
        }
        double x = std::any_cast<double>(args[0]);
        double y = std::any_cast<double>(args[1]);
        return std::pow(x, y);
    }
    
    if (name == "sin") {
        if (args.size() != 1) {
            throw std::runtime_error("sin() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::sin(x);
    }
    
    if (name == "cos") {
        if (args.size() != 1) {
            throw std::runtime_error("cos() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::cos(x);
    }
    
    if (name == "round") {
        if (args.size() != 1) {
            throw std::runtime_error("round() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::round(x);
    }
    
    // ============================================================================
    // Statistical Functions
    // ============================================================================
    if (name == "mean") {
        if (args.empty()) {
            throw std::runtime_error("mean() requires at least 1 argument");
        }
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        return sum / args.size();
    }
    
    if (name == "median") {
        if (args.empty()) {
            throw std::runtime_error("median() requires at least 1 argument");
        }
        std::vector<double> values;
        for (const auto& arg : args) {
            values.push_back(std::any_cast<double>(arg));
        }
        std::sort(values.begin(), values.end());
        
        size_t n = values.size();
        if (n % 2 == 0) {
            return (values[n/2 - 1] + values[n/2]) / 2.0;
        } else {
            return values[n/2];
        }
    }
    
    if (name == "std_dev") {
        if (args.size() < 2) {
            throw std::runtime_error("std_dev() requires at least 2 arguments");
        }
        
        // Calculate mean
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        double mean = sum / args.size();
        
        // Calculate sum of squared differences
        double sum_sq_diff = 0.0;
        for (const auto& arg : args) {
            double x = std::any_cast<double>(arg);
            double diff = x - mean;
            sum_sq_diff += diff * diff;
        }
        
        // Use sample standard deviation (divide by n-1)
        return std::sqrt(sum_sq_diff / (args.size() - 1));
    }
    
    throw std::runtime_error("Unknown function: " + name);
}

bool Interpreter::is_truthy(const Value& value) {
    try {
        return std::any_cast<bool>(value);
    } catch (const std::bad_any_cast&) {
        try {
            double num = std::any_cast<double>(value);
            return num != 0.0;
        } catch (const std::bad_any_cast&) {
            try {
                std::string str = std::any_cast<std::string>(value);
                return !str.empty();
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
    }
}

bool Interpreter::is_equal(const Value& a, const Value& b) {
    // Simplified equality check
    try {
        double a_num = std::any_cast<double>(a);
        double b_num = std::any_cast<double>(b);
        return a_num == b_num;
    } catch (const std::bad_any_cast&) {
        try {
            std::string a_str = std::any_cast<std::string>(a);
            std::string b_str = std::any_cast<std::string>(b);
            return a_str == b_str;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
}

std::string Interpreter::stringify(const Value& value) {
    try {
        return std::to_string(std::any_cast<double>(value));
    } catch (const std::bad_any_cast&) {
        try {
            return std::any_cast<std::string>(value);
        } catch (const std::bad_any_cast&) {
            try {
                bool b = std::any_cast<bool>(value);
                return b ? "true" : "false";
            } catch (const std::bad_any_cast&) {
                try {
                    // Handle arrays
                    auto array = std::any_cast<std::vector<Value>>(value);
                    std::string result = "[";
                    for (size_t i = 0; i < array.size(); ++i) {
                        if (i > 0) result += ", ";
                        result += stringify(array[i]);
                    }
                    result += "]";
                    return result;
                } catch (const std::bad_any_cast&) {
                    return "<unknown value>";
                }
            }
        }
    }
}

Value Interpreter::visit_weighted_sum(const ast::WeightedSum& node) {
   double total = 0.0;
   for (const auto& pair : node.pairs) {
       Value weight_val = evaluate(*pair.first);
       Value value_val = evaluate(*pair.second);
       
       double weight = std::any_cast<double>(weight_val);
       double value = std::any_cast<double>(value_val);
       
       total += weight * value;
   }
   return total;
}

void Interpreter::visit_evolve_statement(const ast::EvolveStatement& node) {
    Value condition_result = evaluate(*node.condition);
    
    if (is_truthy(condition_result)) {
        // Create a new environment for the evolve block
        Environment evolve_env(environment_);
        Environment* previous = environment_;
        environment_ = &evolve_env;
        
        // Execute the evolve block
        for (const auto& stmt : node.body) {
            execute(*stmt);
        }
        
        // Restore previous environment
        environment_ = previous;
    }
}

void Interpreter::visit_if_statement(const ast::IfStatement& node) {
    Value condition_result = evaluate(*node.condition);
    
    if (is_truthy(condition_result)) {
        // Create a new environment for the then branch
        Environment then_env(environment_);
        Environment* previous = environment_;
        environment_ = &then_env;
        
        // Execute the then branch
        for (const auto& stmt : node.then_branch) {
            execute(*stmt);
        }
        
        // Restore previous environment
        environment_ = previous;
    } else if (!node.else_branch.empty()) {
        // Create a new environment for the else branch
        Environment else_env(environment_);
        Environment* previous = environment_;
        environment_ = &else_env;
        
        // Execute the else branch
        for (const auto& stmt : node.else_branch) {
            execute(*stmt);
        }
        
        // Restore previous environment
        environment_ = previous;
    }
}

void Interpreter::visit_while_statement(const ast::WhileStatement& node) {
    // Create a new environment for the while block
    Environment while_env(environment_);
    Environment* previous = environment_;
    environment_ = &while_env;
    
    // Execute the while loop
    while (is_truthy(evaluate(*node.condition))) {
        for (const auto& stmt : node.body) {
            execute(*stmt);
        }
    }
    
    // Restore previous environment
    environment_ = previous;
}

Value UserFunction::call(Interpreter& interpreter, const std::vector<Value>& arguments) {
    // Create a new environment for the function execution
    Environment function_env(closure_);
    Environment* previous = interpreter.environment_;
    interpreter.environment_ = &function_env;

    // Bind arguments to parameters
    for (size_t i = 0; i < declaration_.parameters.size(); i++) {
        const std::string& param_name = declaration_.parameters[i].first;
        // Type checking could be added here in the future using declaration_.parameters[i].second
        if (i < arguments.size()) {
            function_env.define(param_name, arguments[i]);
        } else {
            function_env.define(param_name, nullptr); // Default value for missing args
        }
    }

    try {
        // Execute function body
        for (const auto& stmt : declaration_.body) {
            interpreter.execute(*stmt);
        }
        // If no return statement was encountered, return null
        interpreter.environment_ = previous;
        return nullptr;
    } catch (const ReturnException& return_value) {
        // Handle return statement
        interpreter.environment_ = previous;
        return return_value.value();
    }
}

void Interpreter::visit_function_declaration(const ast::FunctionDeclaration& node) {
    auto function = std::make_shared<UserFunction>(node, environment_);
    environment_->define(node.name, function);
}

void Interpreter::visit_return_statement(const ast::ReturnStatement& node) {
    Value value = node.value ? evaluate(*node.value) : nullptr;
    throw ReturnException(value);
}

void Interpreter::visit_import_statement(const ast::ImportStatement& node) {
    std::cout << "Importing module: " << node.module_path << std::endl;
    
    // For now, just implement a basic file-based import system
    // In a full implementation, this would parse and execute the imported file
    // and bring the specified exports into the current environment
    
    if (!node.imports.empty()) {
        std::cout << "Importing specific items: ";
        for (const auto& import : node.imports) {
            std::cout << import << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Importing all exports from module" << std::endl;
    }
    
    // TODO: Implement actual module loading and namespace management
    // This would involve:
    // 1. Parse the module file
    // 2. Execute it in an isolated environment
    // 3. Extract the exported patterns/functions
    // 4. Import them into the current environment
}

void Interpreter::visit_export_statement(const ast::ExportStatement& node) {
    std::cout << "Exporting: ";
    for (const auto& export_name : node.exports) {
        std::cout << export_name << " ";
    }
    std::cout << std::endl;
    
    // TODO: Implement actual export tracking
    // This would involve marking the specified variables/patterns/functions
    // as available for import by other modules
}

// Variable access methods
Value Interpreter::get_global_variable(const std::string& name) {
    return globals_.get(name);
}

bool Interpreter::has_global_variable(const std::string& name) {
    try {
        globals_.get(name);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

const std::unordered_map<std::string, Value>& Interpreter::get_global_variables() const {
    return globals_.getVariables();
}

// Async/await and exception handling implementation

void Interpreter::visit_async_function_declaration(const ast::AsyncFunctionDeclaration& node) {
    // Create an async function object and store it in the environment
    auto async_func = std::make_shared<AsyncFunction>(node, environment_);
    environment_->define(node.name, async_func);
}

Value Interpreter::visit_await_expression(const ast::AwaitExpression& node) {
    // For now, we'll simulate async by just evaluating the expression normally
    // In a real implementation, this would handle async/await semantics
    Value result = evaluate(*node.expression);
    
    // Simulate async delay (for demonstration)
    std::cout << "[ASYNC] Awaiting expression..." << std::endl;
    
    return result;
}

void Interpreter::visit_try_statement(const ast::TryStatement& node) {
    try {
        // Execute try block
        for (const auto& stmt : node.try_body) {
            execute(*stmt);
        }
    }
    catch (const DSLException& e) {
        // Handle DSL exceptions thrown by 'throw' statements
        if (!node.catch_variable.empty()) {
            environment_->define(node.catch_variable, e.value());
        }
        
        // Execute catch block
        for (const auto& stmt : node.catch_body) {
            execute(*stmt);
        }
    }
    catch (const std::exception& e) {
        // Handle other exceptions
        if (!node.catch_variable.empty()) {
            environment_->define(node.catch_variable, std::string(e.what()));
        }
        
        // Execute catch block
        for (const auto& stmt : node.catch_body) {
            execute(*stmt);
        }
    }
    
    // Execute finally block if present
    if (!node.finally_body.empty()) {
        for (const auto& stmt : node.finally_body) {
            execute(*stmt);
        }
    }
}

void Interpreter::visit_throw_statement(const ast::ThrowStatement& node) {
    Value exception_value = evaluate(*node.expression);
    throw DSLException(exception_value);
}

// AsyncFunction implementation
Value AsyncFunction::call(Interpreter& interpreter, const std::vector<Value>& arguments) {
    // Create a new environment for the function execution
    Environment function_env(closure_);
    Environment* previous = interpreter.environment_;
    interpreter.environment_ = &function_env;

    std::cout << "[ASYNC] Starting async function '" << declaration_.name << "'" << std::endl;

    // Bind arguments to parameters
    for (size_t i = 0; i < declaration_.parameters.size(); i++) {
        const std::string& param_name = declaration_.parameters[i].first;
        // Type checking could be added here in the future using declaration_.parameters[i].second
        if (i < arguments.size()) {
            function_env.define(param_name, arguments[i]);
        } else {
            function_env.define(param_name, nullptr); // Default value for missing args
        }
    }

    try {
        // Execute function body
        for (const auto& stmt : declaration_.body) {
            interpreter.execute(*stmt);
        }
        // If no return statement was encountered, return null
        interpreter.environment_ = previous;
        std::cout << "[ASYNC] Async function '" << declaration_.name << "' completed" << std::endl;
        return nullptr;
    } catch (const ReturnException& return_value) {
        // Handle return statement
        interpreter.environment_ = previous;
        std::cout << "[ASYNC] Async function '" << declaration_.name << "' completed with return" << std::endl;
        return return_value.value();
    }
}

Value Interpreter::visit_array_literal(const ast::ArrayLiteral& node) {
    std::vector<Value> elements;
    elements.reserve(node.elements.size());
    
    for (const auto& element : node.elements) {
        elements.push_back(evaluate(*element));
    }
    
    return elements;
}

Value Interpreter::visit_array_access(const ast::ArrayAccess& node) {
    Value array_value = evaluate(*node.array);
    Value index_value = evaluate(*node.index);
    
    // Try to cast to vector of Values (our array type)
    try {
        auto array = std::any_cast<std::vector<Value>>(array_value);
        auto index = std::any_cast<double>(index_value);
        
        // Convert double index to integer
        size_t idx = static_cast<size_t>(index);
        
        // Check bounds
        if (idx >= array.size()) {
            throw std::runtime_error("Array index " + std::to_string(idx) + " out of bounds (size: " + std::to_string(array.size()) + ")");
        }
        
        return array[idx];
    } catch (const std::bad_any_cast&) {
        throw std::runtime_error("Cannot index into non-array value");
    }
}

} // namespace dsl::runtime
