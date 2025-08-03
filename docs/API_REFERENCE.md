# SEP DSL API Reference

## Overview

This document provides comprehensive documentation for all built-in functions available in the SEP DSL (Sourcegraph Enhanced Pattern Domain Specific Language). Functions are organized by category for easy reference.

## AGI Engine Functions

### measure_coherence(pattern: string) → number
**Real CUDA/Quantum Engine Integration**

Measures the quantum coherence of a given pattern using the QFH (Quantum Field Harmonics) processor.

- **Parameters**: 
  - `pattern` (string): Pattern identifier or data to analyze
- **Returns**: Coherence value between 0.0 and 1.0
- **Engine**: QFHBasedProcessor with GPU acceleration
- **Async Support**: ✅ Can be used with `await`

```sep
coherence = await measure_coherence("sensor_001")
if (coherence > 0.8) {
    print("High coherence detected:", coherence)
}
```

### qfh_analyze(bitstream: string) → number
**Real CUDA/Quantum Engine Integration**

Performs Quantum Field Harmonics analysis on a bitstream with trajectory damping.

- **Parameters**: 
  - `bitstream` (string): Binary data string for analysis
- **Returns**: Rupture ratio between 0.0 and 1.0
- **Engine**: QFHBasedProcessor with exponential damping
- **Async Support**: ✅ Can be used with `await`

```sep
bitstream = extract_bits("sensor_data")
rupture = await qfh_analyze(bitstream)
```

### measure_entropy(pattern: string) → number
**Real Engine Integration**

Calculates Shannon entropy of a given pattern using the pattern analysis engine.

- **Parameters**: 
  - `pattern` (string): Pattern identifier to analyze
- **Returns**: Entropy value between 0.0 and 1.0
- **Engine**: PatternAnalysisEngine
- **Async Support**: ✅ Can be used with `await`

```sep
entropy = await measure_entropy("time_series_data")
if (entropy > 0.9) {
    print("High entropy - potential anomaly")
}
```

### extract_bits(pattern: string) → string
**Real Engine Integration**

Extracts binary representation from pattern data using the bit extraction engine.

- **Parameters**: 
  - `pattern` (string): Pattern identifier for bit extraction
- **Returns**: Binary string representation (e.g., "101010...")
- **Engine**: BitExtractionEngine
- **Async Support**: ✅ Can be used with `await`

```sep
bits = await extract_bits("sensor_pattern")
analysis = await qfh_analyze(bits)
```

### manifold_optimize(pattern: string, coherence: number, stability: number) → number
**Real Engine Integration**

Performs quantum manifold optimization with specified coherence and stability parameters.

- **Parameters**: 
  - `pattern` (string): Pattern identifier
  - `coherence` (number): Target coherence level (0.0-1.0)
  - `stability` (number): Target stability level (0.0-1.0)
- **Returns**: Optimized coherence value
- **Engine**: QuantumManifoldOptimizer
- **Async Support**: ✅ Can be used with `await`

```sep
optimized = await manifold_optimize("pat", 0.8, 0.9)
```

## Core Mathematical Functions

### Basic Operations

#### abs(x: number) → number
Returns the absolute value of x.

```sep
result = abs(-5.2)  // Returns 5.2
```

#### sqrt(x: number) → number
Returns the square root of x. Throws error if x < 0.

```sep
result = sqrt(16)  // Returns 4.0
```

#### min(x1: number, x2: number, ...) → number
Returns the minimum value from all arguments.

```sep
result = min(3, 7, 1, 9)  // Returns 1
```

#### max(x1: number, x2: number, ...) → number
Returns the maximum value from all arguments.

```sep
result = max(3, 7, 1, 9)  // Returns 9
```

### Trigonometric Functions

#### sin(x: number) → number
Returns the sine of x (x in radians).

```sep
result = sin(pi() / 2)  // Returns 1.0
```

#### cos(x: number) → number
Returns the cosine of x (x in radians).

```sep
result = cos(0)  // Returns 1.0
```

#### tan(x: number) → number
Returns the tangent of x (x in radians).

```sep
result = tan(pi() / 4)  // Returns 1.0
```

#### asin(x: number) → number
Returns the arcsine of x. Domain: [-1, 1].

```sep
result = asin(0.5)  // Returns π/6
```

#### acos(x: number) → number
Returns the arccosine of x. Domain: [-1, 1].

```sep
result = acos(0.5)  // Returns π/3
```

#### atan(x: number) → number
Returns the arctangent of x.

```sep
result = atan(1)  // Returns π/4
```

#### atan2(y: number, x: number) → number
Returns the arctangent of y/x, using the signs of both arguments to determine the quadrant.

```sep
result = atan2(1, 1)  // Returns π/4
```

### Exponential and Logarithmic Functions

#### exp(x: number) → number
Returns e^x.

```sep
result = exp(1)  // Returns e ≈ 2.718
```

#### log(x: number) → number
Returns the natural logarithm of x. Throws error if x ≤ 0.

```sep
result = log(e())  // Returns 1.0
```

#### log10(x: number) → number
Returns the base-10 logarithm of x. Throws error if x ≤ 0.

```sep
result = log10(100)  // Returns 2.0
```

#### log2(x: number) → number
Returns the base-2 logarithm of x. Throws error if x ≤ 0.

```sep
result = log2(8)  // Returns 3.0
```

### Power Functions

#### pow(base: number, exponent: number) → number
Returns base^exponent.

```sep
result = pow(2, 3)  // Returns 8.0
```

#### cbrt(x: number) → number
Returns the cube root of x.

```sep
result = cbrt(27)  // Returns 3.0
```

### Rounding Functions

#### floor(x: number) → number
Returns the largest integer ≤ x.

```sep
result = floor(3.7)  // Returns 3.0
```

#### ceil(x: number) → number
Returns the smallest integer ≥ x.

```sep
result = ceil(3.2)  // Returns 4.0
```

#### round(x: number) → number
Returns x rounded to the nearest integer.

```sep
result = round(3.6)  // Returns 4.0
```

#### trunc(x: number) → number
Returns the integer part of x (truncates toward zero).

```sep
result = trunc(3.9)  // Returns 3.0
result = trunc(-3.9)  // Returns -3.0
```

#### fmod(x: number, y: number) → number
Returns the floating-point remainder of x/y. Throws error if y = 0.

```sep
result = fmod(7.5, 2.3)  // Returns remainder
```

### Mathematical Constants

#### pi() → number
Returns the value of π (3.14159...).

```sep
circumference = 2 * pi() * radius
```

#### e() → number
Returns the value of e (2.71828...).

```sep
exponential_growth = initial * pow(e(), rate * time)
```

## Statistical Functions

### Basic Statistics

#### mean(x1: number, x2: number, ...) → number
Returns the arithmetic mean of all arguments.

```sep
average = mean(1, 2, 3, 4, 5)  // Returns 3.0
```

#### sum(x1: number, x2: number, ...) → number
Returns the sum of all arguments.

```sep
total = sum(1, 2, 3, 4, 5)  // Returns 15.0
```

#### count(x1: any, x2: any, ...) → number
Returns the number of arguments passed.

```sep
num_items = count(1, 2, 3, "a", "b")  // Returns 5.0
```

#### median(x1: number, x2: number, ...) → number
Returns the median value of all arguments.

```sep
middle = median(1, 3, 2, 5, 4)  // Returns 3.0
```

### Advanced Statistics

#### stddev(x1: number, x2: number, ...) → number
Returns the sample standard deviation. Requires at least 2 arguments.

```sep
deviation = stddev(1, 2, 3, 4, 5)  // Returns sample std dev
```

#### variance(x1: number, x2: number, ...) → number
Returns the sample variance. Requires at least 2 arguments.

```sep
var = variance(1, 2, 3, 4, 5)  // Returns sample variance
```

#### percentile(p: number, x1: number, x2: number, ...) → number
Returns the p-th percentile of the data. p must be between 0 and 100.

```sep
q75 = percentile(75, 1, 2, 3, 4, 5)  // Returns 75th percentile
```

#### correlation(x1: number, ..., xn: number, y1: number, ..., yn: number) → number
Returns the Pearson correlation coefficient between two datasets. Expects pairs of values (even number of arguments).

```sep
// Correlation between [1,2,3] and [2,4,6]
corr = correlation(1, 2, 3, 2, 4, 6)  // Returns 1.0 (perfect correlation)
```

## Time Series Functions

### Moving Averages

#### moving_average(window_size: number, x1: number, x2: number, ...) → array
Calculates moving averages with the specified window size.

- **Parameters**: 
  - `window_size` (number): Size of the moving window (must be positive)
  - Data values: Numeric data points
- **Returns**: Array of moving averages

```sep
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ma3 = moving_average(3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
// Returns [2, 3, 4, 5, 6, 7, 8, 9] (3-period moving averages)
```

#### exponential_moving_average(alpha: number, initial: number, x1: number, x2: number, ...) → array
Calculates exponential moving averages.

- **Parameters**: 
  - `alpha` (number): Smoothing factor between 0 and 1
  - `initial` (number): Initial EMA value
  - Data values: Numeric data points
- **Returns**: Array of EMA values

```sep
ema = exponential_moving_average(0.3, 10, 12, 11, 13, 15, 14)
// Returns EMA sequence starting from initial value
```

### Trend Analysis

#### trend_detection(threshold: number, x1: number, x2: number, ...) → number
Detects linear trends in time series data using least squares regression.

- **Parameters**: 
  - `threshold` (number): Minimum slope magnitude to consider significant
  - Data values: Time series data points
- **Returns**: 1 (upward trend), -1 (downward trend), 0 (no trend)

```sep
trend = trend_detection(0.1, 1, 2, 3, 4, 5)  // Returns 1 (upward)
```

#### rate_of_change(x1: number, x2: number, ...) → array
Calculates the rate of change between consecutive data points.

- **Parameters**: Data values (numeric time series)
- **Returns**: Array of rate of change values

```sep
roc = rate_of_change(10, 12, 11, 15, 13)
// Returns [0.2, -0.083, 0.36, -0.133] (percentage changes)
```

## Data Transformation Functions

### Normalization

#### normalize(x1: number, x2: number, ...) → array
Normalizes data to the range [0, 1] using min-max scaling.

```sep
normalized = normalize(1, 5, 3, 9, 2)
// Returns data scaled to [0, 1] range
```

#### standardize(x1: number, x2: number, ...) → array
Standardizes data to have mean=0 and standard deviation=1 (z-score normalization).

```sep
standardized = standardize(1, 5, 3, 9, 2)
// Returns z-score normalized data
```

#### scale(factor: number, x1: number, x2: number, ...) → array
Scales all values by a constant factor.

- **Parameters**: 
  - `factor` (number): Scaling factor
  - Data values: Numeric data to scale

```sep
scaled = scale(2.5, 1, 2, 3, 4)
// Returns [2.5, 5.0, 7.5, 10.0]
```

### Filtering

#### filter_above(threshold: number, x1: number, x2: number, ...) → array
Filters values above a threshold (keeps values > threshold).

```sep
filtered = filter_above(3, 1, 5, 2, 8, 3, 6)
// Returns [5, 8, 6] (values > 3)
```

#### filter_below(threshold: number, x1: number, x2: number, ...) → array
Filters values below a threshold (keeps values < threshold).

```sep
filtered = filter_below(5, 1, 5, 2, 8, 3, 6)
// Returns [1, 2, 3] (values < 5)
```

#### filter_range(min: number, max: number, x1: number, x2: number, ...) → array
Filters values within a specified range [min, max] (inclusive).

```sep
filtered = filter_range(2, 6, 1, 5, 2, 8, 3, 6)
// Returns [5, 2, 3, 6] (values between 2 and 6)
```

#### clamp(min: number, max: number, x1: number, x2: number, ...) → array
Clamps values to a specified range [min, max].

```sep
clamped = clamp(2, 6, 1, 5, 2, 8, 3, 6)
// Returns [2, 5, 2, 6, 3, 6] (values clamped to [2, 6])
```

## Pattern Matching Functions

### Regular Expressions

#### regex_match(pattern: string, text: string) → boolean
Tests if text matches the regex pattern.

```sep
matches = regex_match("\\d+", "abc123def")  // Returns true
```

#### regex_extract(pattern: string, text: string) → string
Extracts the first match of the regex pattern from text.

```sep
number = regex_extract("\\d+", "abc123def")  // Returns "123"
```

#### regex_replace(pattern: string, replacement: string, text: string) → string
Replaces all matches of the regex pattern in text with replacement.

```sep
clean = regex_replace("\\d+", "X", "abc123def456")  // Returns "abcXdefX"
```

### Fuzzy Matching

#### fuzzy_match(text1: string, text2: string, threshold: number) → boolean
Tests if two strings are similar within the specified threshold using Levenshtein distance.

- **Parameters**: 
  - `text1`, `text2` (string): Strings to compare
  - `threshold` (number): Similarity threshold (0.0-1.0)

```sep
similar = fuzzy_match("hello", "helo", 0.8)  // Returns true
```

#### fuzzy_similarity(text1: string, text2: string) → number
Returns the similarity score between two strings (0.0-1.0).

```sep
score = fuzzy_similarity("kitten", "sitting")  // Returns similarity score
```

## Aggregation Functions

### Data Grouping

#### groupby(key_func: string, x1: any, x2: any, ...) → object
Groups data by a key function (simplified string-based grouping).

```sep
grouped = groupby("type", "a", "b", "a", "c", "b")
// Groups items by their value
```

#### pivot(rows: array, columns: array, values: array) → object
Creates a pivot table from row, column, and value arrays.

```sep
pivoted = pivot(["A", "B"], ["X", "Y"], [10, 20])
// Creates pivot table structure
```

#### rollup(operation: string, x1: number, x2: number, ...) → number
Performs rollup aggregation with specified operation.

- **Parameters**: 
  - `operation` (string): "sum", "mean", "max", "min", "count"
  - Data values: Numeric data to aggregate

```sep
total = rollup("sum", 1, 2, 3, 4, 5)  // Returns 15
average = rollup("mean", 1, 2, 3, 4, 5)  // Returns 3
```

#### aggregate(operations: array, x1: number, x2: number, ...) → object
Performs multiple aggregation operations at once.

```sep
stats = aggregate(["sum", "mean", "max"], 1, 2, 3, 4, 5)
// Returns object with sum, mean, and max values
```

## Array Support

### Array Literals
Create arrays using square bracket notation:

```sep
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, true]
nested = [[1, 2], [3, 4], [5, 6]]
```

### Array Access
Access array elements using bracket notation:

```sep
arr = [10, 20, 30, 40]
first = arr[0]    // Returns 10
second = arr[1]   // Returns 20
last = arr[3]     // Returns 40
```

### Array Operations
Many built-in functions return arrays and can work with array inputs:

```sep
data = [1, 2, 3, 4, 5]
ma = moving_average(3, data[0], data[1], data[2], data[3], data[4])
normalized = normalize(data[0], data[1], data[2], data[3], data[4])
```

## Error Handling

### Exception Handling
All built-in functions throw descriptive runtime errors for invalid inputs:

```sep
pattern robust_calculation {
    try {
        result = sqrt(-1)  // Will throw error
    }
    catch (error) {
        print("Error:", error)
        result = 0
    }
    finally {
        print("Calculation completed")
    }
}
```

### Common Error Types
- **Domain errors**: Invalid input ranges (e.g., `sqrt(-1)`, `log(0)`)
- **Argument errors**: Wrong number of arguments
- **Type errors**: Invalid argument types
- **Mathematical errors**: Division by zero, overflow

## Performance Notes

### CUDA Acceleration
AGI engine functions (`measure_coherence`, `qfh_analyze`, etc.) use GPU acceleration when available:

```sep
// These calls use CUDA/quantum processors
entropy = await measure_entropy("large_dataset")
coherence = await measure_coherence("complex_pattern")
```

### Async/Await Support
Engine functions support asynchronous execution:

```sep
async function analyzePattern(data) {
    entropy = await measure_entropy(data)
    coherence = await measure_coherence(data)
    return entropy + coherence
}

pattern analysis {
    result = await analyzePattern("sensor_data")
}
```

### Memory Considerations
- Array operations create new arrays (immutable)
- Large datasets should use streaming when possible
- Statistical functions process all arguments in memory

## Batch Processing Functions

### process_batch(pattern_ids: array, pattern_codes: array, [inputs], [max_threads], [batch_size], [fail_fast], [timeout]) → object
**Advanced Parallel Processing**

Processes multiple patterns in parallel with configurable threading and performance options.

- **Parameters**:
  - `pattern_ids` (array): Array of pattern identifiers
  - `pattern_codes` (array): Array of pattern code strings to execute
  - `inputs` (array, optional): Array of input variable bindings per pattern
  - `max_threads` (number, optional): Maximum parallel threads (default: auto-detect)
  - `batch_size` (number, optional): Batch size for processing (default: 100)
  - `fail_fast` (boolean, optional): Stop on first error (default: false)
  - `timeout` (number, optional): Timeout in seconds (default: 30.0)
- **Returns**: Object with batch results and statistics
- **Engine**: BatchProcessor with configurable parallelism

```sep
pattern_ids = ["sensor_1", "sensor_2", "sensor_3"]
pattern_codes = [
    "measure_coherence(\"data_1\")",
    "measure_entropy(\"data_2\")",
    "qfh_analyze(\"101010\")"
]

result = process_batch(pattern_ids, pattern_codes, null, 4, 10, false, 30.0)
print("Processed:", result["patterns_processed"])
print("Succeeded:", result["patterns_succeeded"])
print("Total time:", result["total_time_ms"], "ms")

// Access individual results
for (i = 0; i < len(result["results"]); i = i + 1) {
    pattern_result = result["results"][i]
    print("Pattern", pattern_result["pattern_id"] + ":", pattern_result["value"])
}
```

## Engine Configuration Functions

### set_engine_config(parameter_name: string, value_type: string, value_string: string) → boolean
**Runtime Engine Tuning**

Sets an engine configuration parameter at runtime.

- **Parameters**:
  - `parameter_name` (string): Configuration parameter name (e.g., "quantum.coherence_threshold")
  - `value_type` (string): Value type ("bool", "int", "double", "string")
  - `value_string` (string): String representation of the value
- **Returns**: true if successfully set, false otherwise
- **Engine**: EngineConfig with validation

```sep
// Adjust quantum processing threshold
success = set_engine_config("quantum.coherence_threshold", "double", "0.8")
print("Config updated:", success)

// Enable debug logging
set_engine_config("debug.log_level", "int", "4")

// Configure batch processing defaults
set_engine_config("batch.default_max_threads", "int", "8")
```

### get_engine_config(parameter_name: string) → object
**Configuration Value Retrieval**

Gets the current value of an engine configuration parameter.

- **Parameters**:
  - `parameter_name` (string): Configuration parameter name
- **Returns**: Object with parameter name, type, and current value
- **Engine**: EngineConfig

```sep
coherence_config = get_engine_config("quantum.coherence_threshold")
print("Current threshold:", coherence_config["value_string"])
print("Value type:", coherence_config["value_type"])

// Check if GPU is enabled
gpu_config = get_engine_config("cuda.enable_gpu")
if (gpu_config["value_string"] == "true") {
    print("GPU acceleration is enabled")
}
```

### list_engine_config() → array
**Configuration Parameter Discovery**

Lists all available engine configuration parameters with their descriptions.

- **Parameters**: None
- **Returns**: Array of objects with parameter information
- **Engine**: EngineConfig

```sep
config_list = list_engine_config()
print("Available parameters:", len(config_list))

// Show quantum parameters
for (i = 0; i < len(config_list); i = i + 1) {
    param = config_list[i]
    if (param["category"] == "quantum") {
        print("Parameter:", param["name"])
        print("  Description:", param["description"])
        print("  Requires restart:", param["requires_restart"])
    }
}
```

### reset_engine_config([category: string]) → boolean
**Configuration Reset**

Resets engine configuration to defaults, optionally for a specific category.

- **Parameters**:
  - `category` (string, optional): Configuration category to reset ("quantum", "cuda", "memory", etc.)
- **Returns**: true if successfully reset
- **Engine**: EngineConfig

```sep
// Reset all configuration
reset_engine_config()

// Reset only quantum parameters
reset_engine_config("quantum")

// Reset performance settings
reset_engine_config("performance")
```

## Testing & Quality Assurance

SEP DSL includes comprehensive testing infrastructure for production-grade reliability:

### Fuzz Testing Commands

```bash
# Quick robustness testing (30 seconds each)
./run_fuzz_tests.sh quick

# Comprehensive testing (5 minutes each)
./run_fuzz_tests.sh comprehensive

# Manual fuzzing with custom duration
./run_fuzz_docker.sh parser 3600      # 1 hour parser fuzzing
./run_fuzz_docker.sh interpreter 1800 # 30 min interpreter fuzzing
```

### Validation Commands

```bash
# Complete test suite validation
./build.sh

# DSL-specific tests
./build/tests/dsl_interpreter_test
./build/tests/dsl_parser_test

# Performance benchmarks
./build/examples/pattern_metric_example --benchmark
```

### Quality Assurance Features

- **LibFuzzer Integration**: Coverage-guided fuzzing with AddressSanitizer
- **Docker-based Testing**: Consistent testing environment
- **Memory Safety**: Detection of buffer overflows and corruption
- **Crash Detection**: Automatic discovery of parser/interpreter bugs
- **Corpus Management**: Seeded with realistic DSL programs

## Version Information

**Current Version**: 1.2.0  
**Last Updated**: August 2025  
**Engine Integration**: Production-ready with CUDA acceleration and advanced batch processing  
**Total Built-in Functions**: 70+
**Quality Assurance**: LibFuzzer integration with comprehensive fuzz testing

---

*This documentation covers all built-in functions available in SEP DSL v1.2.0. For examples and tutorials, see the [Language Reference Manual](LANGUAGE_REFERENCE.md) and [Getting Started Guide](../README.md).*
