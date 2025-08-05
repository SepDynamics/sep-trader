# SEP DSL Language Reference Manual

**Version 1.2.0** | **August 2025**

## Table of Contents

1. [Introduction](#introduction)
2. [Lexical Elements](#lexical-elements)
3. [Types and Values](#types-and-values)
4. [Expressions](#expressions)
5. [Statements](#statements)
6. [Declarations](#declarations)
7. [Built-in Functions](#built-in-functions)
8. [Pattern System](#pattern-system)
9. [Async Programming](#async-programming)
10. [Exception Handling](#exception-handling)
11. [Examples](#examples)

## Introduction

The **SEP DSL** (Sourcegraph Engineering Platform Domain-Specific Language) is a specialized language designed for quantum pattern analysis and AGI-based coherence measurement. This reference manual provides complete documentation of the language syntax, semantics, and built-in capabilities.

### Design Goals

- **Quantum-native**: Direct integration with quantum field harmonics and coherence analysis
- **Performance**: CUDA-accelerated execution with minimal overhead
- **Expressiveness**: Rich pattern matching and data transformation capabilities
- **Safety**: Strong type hints, exception handling, and memory safety

## Lexical Elements

### Keywords

The following identifiers are reserved keywords in SEP DSL:

```
pattern     stream      signal      memory      from        when        using
input       output      evolve      inherits    if          else        while  
for         function    return      break       continue    import      export
async       await       try         catch       throw       finally     true
false       null        number      string      bool        pattern_type void
```

### Identifiers

Identifiers follow standard programming language conventions:
- Must start with a letter or underscore
- Can contain letters, digits, and underscores
- Case-sensitive
- Must not be a reserved keyword

```sep
my_variable     // Valid
_private_var    // Valid
Variable123     // Valid
123invalid      // Invalid - starts with digit
pattern         // Invalid - reserved keyword
```

### Literals

#### Number Literals
```sep
42              // Integer
3.14159         // Floating point
-7.5            // Negative number
1.23e-4         // Scientific notation
```

#### String Literals
```sep
"hello world"   // Basic string
"with \"quotes\"" // Escaped quotes
"multiline
string"         // Multiline strings supported
```

#### Boolean Literals
```sep
true            // Boolean true
false           // Boolean false
```

#### Array Literals
```sep
[1, 2, 3]                    // Number array
["a", "b", "c"]              // String array
[true, false, true]          // Boolean array
[1, "mixed", true]           // Mixed type array
[]                           // Empty array
```

### Operators

#### Arithmetic Operators
| Operator | Description | Precedence | Associativity |
|----------|-------------|------------|---------------|
| `+` | Addition | 10 | Left |
| `-` | Subtraction | 10 | Left |
| `*` | Multiplication | 11 | Left |
| `/` | Division | 11 | Left |
| `-` | Unary minus | 13 | Right |

#### Comparison Operators
| Operator | Description | Precedence | Associativity |
|----------|-------------|------------|---------------|
| `<` | Less than | 8 | Left |
| `<=` | Less than or equal | 8 | Left |
| `>` | Greater than | 8 | Left |
| `>=` | Greater than or equal | 8 | Left |
| `==` | Equal | 7 | Left |
| `!=` | Not equal | 7 | Left |

#### Logical Operators
| Operator | Description | Precedence | Associativity |
|----------|-------------|------------|---------------|
| `||` | Logical OR | 2 | Left |
| `&&` | Logical AND | 3 | Left |
| `!` | Logical NOT | 13 | Right |

#### Access Operators
| Operator | Description | Precedence | Associativity |
|----------|-------------|------------|---------------|
| `[]` | Array access | 14 | Left |
| `.` | Member access | 16 | Left |
| `()` | Function call | 15 | Left |

## Types and Values

### Basic Types

#### Number
Represents both integers and floating-point numbers.
```sep
temperature: number = 98.6
count: number = 42
```

#### String
Text data with UTF-8 encoding.
```sep
message: string = "Hello, World!"
name: string = "SEP DSL"
```

#### Boolean
Logical true/false values.
```sep
is_active: bool = true
has_errors: bool = false
```

#### Array
Ordered collection of values, supporting mixed types.
```sep
numbers: [number] = [1, 2, 3, 4, 5]
mixed: [] = [1, "text", true, 3.14]
```

### Type Annotations

Type annotations are optional but recommended for clarity and better error messages:

```sep
function calculate_entropy(data: [number]): number {
    // Function implementation
    return 0.5
}

pattern data_analysis {
    sensor_data: [number] = [1.2, 3.4, 5.6]
    result: number = calculate_entropy(sensor_data)
}
```

## Expressions

### Primary Expressions

#### Literals
```sep
42              // Number literal
"text"          // String literal
true            // Boolean literal
[1, 2, 3]       // Array literal
```

#### Identifiers
```sep
variable_name   // Variable reference
function_call() // Function call
```

#### Parenthesized Expressions
```sep
(2 + 3) * 4     // Grouping for precedence
```

### Binary Expressions

Binary expressions follow strict precedence rules:

```sep
// Arithmetic (left-associative)
result = 2 + 3 * 4      // Equivalent to: 2 + (3 * 4) = 14
mixed = (2 + 3) * 4     // Explicit grouping: (2 + 3) * 4 = 20

// Comparison
comparison = 5 > 3 && 2 < 4    // Equivalent to: (5 > 3) && (2 < 4) = true

// Logical
logic = true || false && false  // Equivalent to: true || (false && false) = true
```

### Unary Expressions

```sep
negated = -42           // Unary minus
inverted = !true        // Logical NOT
```

### Array Access

```sep
arr = [10, 20, 30, 40]
first = arr[0]          // Access first element: 10
last = arr[3]           // Access last element: 40
```

### Function Calls

```sep
// Built-in function calls
entropy_val = measure_entropy("pattern_data")
sine_val = sin(3.14159 / 2)

// User-defined function calls
result = my_function(arg1, arg2, arg3)
```

## Statements

### Expression Statements

Any expression can be used as a statement:

```sep
calculate_result()      // Function call statement
x = 5                   // Assignment statement
x + y                   // Expression statement (computed but not stored)
```

### Variable Declarations

```sep
// Basic declaration with inference
temperature = 98.6

// Declaration with type annotation
sensor_id: string = "temp_01"

// Array declaration
measurements: [number] = [98.6, 99.1, 97.8]
```

### Control Flow

#### If Statements
```sep
if (temperature > 100) {
    status = "overheating"
} else if (temperature > 80) {
    status = "warm"
} else {
    status = "normal"
}
```

#### While Loops
```sep
counter = 0
while (counter < 10) {
    process_data(counter)
    counter = counter + 1
}
```

### Print Statements

```sep
print("Simple message")
print("Temperature:", temperature)
print("Status:", status, "at", current_time)
```

## Declarations

### Function Declarations

#### Synchronous Functions
```sep
function calculate_average(values: [number]): number {
    sum = 0
    count = 0
    for value in values {
        sum = sum + value
        count = count + 1
    }
    return sum / count
}
```

#### Async Functions
```sep
async function analyze_sensor_data(sensor_id: string): number {
    raw_data = await fetch_sensor_readings(sensor_id)
    entropy = await measure_entropy(raw_data)
    coherence = await measure_coherence(raw_data)
    return entropy + coherence
}
```

### Pattern Declarations

Patterns are the core construct for defining quantum analysis workflows:

```sep
pattern sensor_monitoring {
    // Pattern variables
    sensor_ids = ["temp_01", "temp_02", "pressure_01"]
    threshold = 0.8
    
    // Analysis logic
    for sensor_id in sensor_ids {
        entropy = measure_entropy(sensor_id)
        if (entropy > threshold) {
            print("Anomaly detected in", sensor_id)
        }
    }
}
```

#### Pattern Inheritance
```sep
pattern base_analysis {
    threshold = 0.5
    
    function is_anomaly(value: number): bool {
        return value > threshold
    }
}

pattern temperature_analysis inherits base_analysis {
    threshold = 0.8  // Override base threshold
    
    sensor_data = measure_entropy("temp_sensor")
    if (is_anomaly(sensor_data)) {
        print("Temperature anomaly detected")
    }
}
```

### Stream Declarations

Streams define data sources for real-time analysis:

```sep
stream sensor_data from "oanda://EUR_USD" {
    timeframe = "M1"
    buffer_size = 1000
}
```

### Signal Declarations

Signals define output conditions and confidence thresholds:

```sep
signal trading_opportunity
    trigger = coherence > 0.7 && entropy < 0.3
    confidence = coherence * (1 - entropy)
```

## Built-in Functions

### Quantum Analysis Functions

#### Core AGI Functions
```sep
// Measure quantum coherence of a pattern
coherence = measure_coherence(pattern_data: string): number

// Calculate Shannon entropy
entropy = measure_entropy(pattern_data: string): number

// Quantum field harmonics analysis
qfh_result = qfh_analyze(bitstream: string): number

// Extract binary representation
bits = extract_bits(pattern: string): string

// Manifold optimization
optimized = manifold_optimize(pattern: string, coherence: number, stability: number): number
```

### Mathematical Functions

#### Core Math
```sep
// Trigonometric functions
sin_val = sin(angle)        cos_val = cos(angle)        tan_val = tan(angle)
asin_val = asin(value)      acos_val = acos(value)      atan_val = atan(value)

// Exponential and logarithmic
exp_val = exp(x)            log_val = log(x)            log10_val = log10(x)
pow_val = pow(base, exp)    sqrt_val = sqrt(x)          abs_val = abs(x)

// Rounding functions
ceil_val = ceil(x)          floor_val = floor(x)        round_val = round(x)
```

### Statistical Functions

```sep
// Central tendency
avg = mean([1, 2, 3, 4, 5])          // Calculate arithmetic mean
mid = median([1, 2, 3, 4, 5])        // Calculate median value

// Variability
std = stddev([1, 2, 3, 4, 5])        // Standard deviation
var = variance([1, 2, 3, 4, 5])      // Variance

// Correlation
corr = correlation([1,2,3], [2,4,6]) // Pearson correlation coefficient

// Percentiles
p90 = percentile([1,2,3,4,5], 90)    // 90th percentile
```

### Time Series Functions

```sep
// Moving averages
ma = moving_average(data, window_size)
ema = exponential_moving_average(data, alpha)

// Trend analysis
trend = trend_detection(data, window_size)
roc = rate_of_change(data, period)
```

### Data Transformation Functions

```sep
// Normalization and scaling
normalized = normalize(data)                    // Scale to [0, 1]
standardized = standardize(data)                // Z-score normalization
scaled = scale(data, min_val, max_val)          // Scale to custom range

// Filtering
above = filter_above(data, threshold)           // Values above threshold
below = filter_below(data, threshold)           // Values below threshold
range_filtered = filter_range(data, min, max)   // Values in range
clamped = clamp(data, min, max)                 // Clamp to range
```

### Pattern Matching Functions

```sep
// Regular expressions
matches = regex_match(text, pattern)            // Boolean match
extracted = regex_extract(text, pattern)        // Extract matches
replaced = regex_replace(text, pattern, replacement)

// Fuzzy matching
is_fuzzy_match = fuzzy_match(text1, text2, threshold)
similarity = fuzzy_similarity(text1, text2)    // Similarity score [0,1]
```

### Aggregation Functions

```sep
// Group operations
grouped = groupby(data, key_function)           // Group by key
pivot_table = pivot(data, rows, cols, values)  // Create pivot table
rolled_up = rollup(data, dimensions, metrics)  // Roll up dimensions

// General aggregation
aggregated = aggregate(data, operation)         // Apply operation to groups
```

## Pattern System

### Basic Patterns

```sep
pattern hello_world {
    message = "Hello, World!"
    print(message)
}
```

### Pattern Variables

```sep
pattern sensor_analysis {
    // Pattern-scoped variables
    sensor_id = "temp_01"
    threshold = 0.8
    
    // Computed values
    current_reading = measure_entropy(sensor_id)
    is_anomaly = current_reading > threshold
    
    print("Sensor", sensor_id, "reading:", current_reading)
    if (is_anomaly) {
        print("ALERT: Anomaly detected!")
    }
}
```

### Pattern Composition

```sep
pattern base_monitoring {
    threshold = 0.5
    
    function check_threshold(value: number): bool {
        return value > threshold
    }
}

pattern temperature_monitoring inherits base_monitoring {
    threshold = 0.8  // Override base value
    
    temp_reading = measure_entropy("temperature_sensor")
    if (check_threshold(temp_reading)) {
        print("Temperature threshold exceeded")
    }
}
```

## Async Programming

### Async Functions

```sep
async function fetch_and_analyze(sensor_id: string): number {
    try {
        // Await asynchronous operations
        raw_data = await fetch_sensor_data(sensor_id)
        entropy = await measure_entropy(raw_data)
        coherence = await measure_coherence(raw_data)
        
        return entropy + coherence
    } catch (error) {
        print("Error processing sensor:", error)
        return -1
    }
}
```

### Async Patterns

```sep
pattern realtime_analysis {
    sensor_ids = ["temp_01", "temp_02", "pressure_01"]
    
    for sensor_id in sensor_ids {
        result = await fetch_and_analyze(sensor_id)
        print("Sensor", sensor_id, "analysis result:", result)
    }
}
```

## Exception Handling

### Try-Catch-Finally

```sep
pattern robust_analysis {
    try {
        data = measure_entropy("sensor_001")
        if (data > 0.8) {
            throw "Anomaly detected in sensor data"
        }
        print("Normal operation: entropy =", data)
    } catch (error) {
        print("Error caught:", error)
        // Error recovery logic
        data = 0.0
    } finally {
        print("Analysis completed at", timestamp())
    }
}
```

### Error Propagation

```sep
async function safe_analysis(sensor_id: string): number {
    try {
        result = await measure_entropy(sensor_id)
        if (result < 0) {
            throw "Invalid entropy measurement"
        }
        return result
    } catch (error) {
        print("Analysis failed:", error)
        throw error  // Re-throw to caller
    }
}
```

## Examples

### Basic Quantum Analysis

```sep
pattern quantum_coherence_example {
    // Define pattern data
    pattern_data = "sensor_reading_stream_001"
    
    // Perform quantum analysis
    coherence = measure_coherence(pattern_data)
    entropy = measure_entropy(pattern_data)
    
    // Calculate stability score
    stability = 1.0 - entropy
    
    // Decision logic
    if (coherence > 0.7 && stability > 0.6) {
        print("High-quality pattern detected")
        print("Coherence:", coherence, "Stability:", stability)
    } else {
        print("Pattern quality insufficient")
    }
}
```

### Advanced Data Processing

```sep
pattern advanced_data_processing {
    // Raw sensor data
    raw_data = [1.2, 2.3, 1.8, 4.5, 3.2, 2.1, 5.8, 3.9]
    
    // Data preprocessing
    normalized_data = normalize(raw_data)
    filtered_data = filter_above(normalized_data, 0.3)
    
    // Statistical analysis
    avg_value = mean(filtered_data)
    std_dev = stddev(filtered_data)
    trend = trend_detection(filtered_data, 3)
    
    // Pattern matching
    anomaly_pattern = "high_variance_.*"
    data_signature = "high_variance_" + std_dev
    
    if (regex_match(data_signature, anomaly_pattern)) {
        print("Anomaly pattern detected in data")
        print("Average:", avg_value, "Std Dev:", std_dev)
        print("Trend:", trend)
    }
}
```

### Async Real-time Monitoring

```sep
async function process_sensor_stream(sensor_id: string): number {
    try {
        // Fetch real-time data
        stream_data = await fetch_live_stream(sensor_id)
        
        // Apply quantum analysis
        coherence = await measure_coherence(stream_data)
        entropy = await measure_entropy(stream_data)
        
        // Compute composite score
        quality_score = coherence * (1.0 - entropy)
        
        return quality_score
    } catch (error) {
        print("Stream processing failed:", error)
        return 0.0
    }
}

pattern realtime_monitoring {
    sensor_list = ["temp_01", "temp_02", "pressure_01", "flow_01"]
    quality_threshold = 0.8
    
    for sensor in sensor_list {
        quality = await process_sensor_stream(sensor)
        
        if (quality > quality_threshold) {
            print("✅ Sensor", sensor, "quality:", quality)
        } else {
            print("⚠️ Sensor", sensor, "below threshold:", quality)
        }
    }
}
```

### Error Handling and Recovery

```sep
pattern fault_tolerant_analysis {
    sensors = ["primary", "backup", "emergency"]
    
    for sensor in sensors {
        try {
            reading = await measure_entropy(sensor)
            
            if (reading > 0.9) {
                throw "Critical entropy level in " + sensor
            }
            
            print("Sensor", sensor, "operating normally:", reading)
            break  // Success, exit loop
            
        } catch (error) {
            print("Error with sensor", sensor, ":", error)
            
            if (sensor == "emergency") {
                print("All sensors failed - entering safe mode")
                throw "System-wide sensor failure"
            }
            
            print("Trying next sensor...")
            continue
            
        } finally {
            print("Completed check for sensor", sensor)
        }
    }
}
```

---

## Appendices

### A. Reserved Keywords Reference

Complete list of reserved keywords in SEP DSL:

```
async       await       bool        break       catch       continue
else        evolve      export      false       finally     for
from        function    if          import      inherits    input
memory      null        number      output      pattern     pattern_type
return      signal      stream      string      throw       true
try         using       void        when        while
```

### B. Operator Precedence Table

| Precedence | Operators | Associativity | Description |
|------------|-----------|---------------|-------------|
| 16 | `.` `->` | Left | Member access |
| 15 | `()` | Left | Function calls |
| 14 | `[]` `++` `--` | Left | Array access, postfix |
| 13 | `!` `-` `+` `~` `++` `--` | Right | Unary operators |
| 12 | `**` | Right | Exponentiation |
| 11 | `*` `/` `%` | Left | Multiplicative |
| 10 | `+` `-` | Left | Additive |
| 9 | `<<` `>>` | Left | Shift |
| 8 | `<` `<=` `>` `>=` | Left | Relational |
| 7 | `==` `!=` | Left | Equality |
| 6 | `&` | Left | Bitwise AND |
| 5 | `^` | Left | Bitwise XOR |
| 4 | `|` | Left | Bitwise OR |
| 3 | `&&` | Left | Logical AND |
| 2 | `||` | Left | Logical OR |
| 1 | `?:` | Right | Ternary conditional |

### C. Built-in Function Reference

See [Built-in Functions](#built-in-functions) section for complete documentation of all available functions organized by category.

---

*This language reference manual is current as of SEP DSL version 1.2.0. For the latest updates and examples, visit the [SEP DSL GitHub repository](https://github.com/scrallex/dsl).*
