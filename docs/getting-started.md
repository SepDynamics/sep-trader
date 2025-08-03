# Getting Started with SEP DSL

Welcome to SEP DSL! This guide will help you get up and running quickly with the AGI Coherence Framework.

## Table of Contents

1. [Installation](#installation)
2. [Your First Program](#your-first-program)
3. [Core Concepts](#core-concepts)
4. [Language Features](#language-features)
5. [AGI Functions](#agi-functions)
6. [Examples Walkthrough](#examples-walkthrough)
7. [Language Bindings](#language-bindings)
8. [Next Steps](#next-steps)

## Installation

### Option 1: Docker (Recommended)

The easiest way to get started:

```bash
# Clone the repository
git clone https://github.com/scrallex/dsl.git
cd dsl

# Build with Docker (includes CUDA support)
./build.sh
```

### Option 2: Local Installation

For development or custom builds:

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install cmake build-essential clang-15 cuda-toolkit

# Clone and build
git clone https://github.com/scrallex/dsl.git
cd dsl
./install.sh --local --no-docker
./build.sh --no-docker
```

### Verify Installation

```bash
# Test the interpreter
echo 'pattern test { print("Hello, SEP DSL!") }' | ./build/src/dsl/sep_dsl_interpreter

# Expected output:
# Hello, SEP DSL!
```

## Your First Program

Create a file called `hello.sep`:

```sep
#!/usr/bin/env /path/to/sep_dsl_interpreter

pattern hello_world {
    // Variables
    name = "Developer"
    value = 42
    
    // AGI engine integration
    data = "sample_data"
    coherence = measure_coherence(data)
    
    // Output
    print("Hello,", name)
    print("Value:", value)
    print("Coherence:", coherence)
}
```

Run it:

```bash
# Make executable
chmod +x hello.sep

# Run directly
./hello.sep

# Or run with interpreter
./build/src/dsl/sep_dsl_interpreter hello.sep
```

## Core Concepts

### Patterns

Patterns are the fundamental building blocks in SEP DSL. They encapsulate related computations and variables:

```sep
pattern data_analysis {
    // Variables are scoped to this pattern
    input_data = "sensor_readings"
    
    // Computations
    coherence = measure_coherence(input_data)
    entropy = measure_entropy(input_data)
    
    // Results
    quality_score = coherence * (1.0 - entropy)
}
```

### Variable Scoping

Variables within patterns are isolated but can be accessed from outside:

```sep
pattern analysis {
    internal_value = 100
    result = internal_value * 2
}

pattern consumer {
    // Access variables from other patterns
    external_result = analysis.result  // This would be 200
}
```

### Real-time Processing

SEP DSL is designed for real-time applications:

```sep
pattern real_time_monitor {
    // In production, these would be live data streams
    sensor_stream = "live_sensor_data"
    
    // Process in real-time
    current_coherence = measure_coherence(sensor_stream)
    
    // React immediately
    if current_coherence < 0.3 {
        print("ALERT: Low coherence detected!")
    }
}
```

## Language Features

### Data Types

```sep
pattern data_types_demo {
    // Numbers (integers and floats)
    integer_val = 42
    float_val = 3.14159
    
    // Strings
    message = "Hello, World!"
    
    // Booleans
    is_active = true
    is_complete = false
    
    // Computed values
    sum = integer_val + float_val
    is_positive = sum > 0
}
```

### Control Flow

```sep
pattern control_flow {
    temperature = 25.5
    
    // If-else statements
    if temperature > 30 {
        status = "Hot"
    } else if temperature > 20 {
        status = "Comfortable"
    } else {
        status = "Cold"
    }
    
    // While loops (where supported)
    counter = 0
    while counter < 5 {
        print("Count:", counter)
        counter = counter + 1
    }
}
```

### Operators

```sep
pattern operators_demo {
    // Arithmetic
    a = 10
    b = 3
    sum = a + b          // 13
    difference = a - b   // 7
    product = a * b      // 30
    quotient = a / b     // 3.333...
    
    // Comparison
    equal = (a == b)     // false
    not_equal = (a != b) // true
    greater = (a > b)    // true
    less_equal = (a <= b) // false
    
    // Logical
    both_true = true && true   // true
    either_true = true || false // true
    negated = !true           // false
}
```

## AGI Functions

SEP DSL's power comes from its built-in AGI analysis functions:

### Core Analysis Functions

```sep
pattern agi_functions {
    data_sample = "time_series_data"
    
    // Quantum coherence analysis (0.0 = chaotic, 1.0 = perfectly ordered)
    coherence = measure_coherence(data_sample)
    
    // Shannon entropy (0.0 = predictable, 1.0 = random)
    entropy = measure_entropy(data_sample)
    
    // Bit pattern extraction
    bit_pattern = extract_bits(data_sample)
    
    // Quantum field harmonics analysis
    qfh_result = qfh_analyze(bit_pattern)
    
    // Pattern optimization
    optimized = manifold_optimize(data_sample, 0.8, 0.9)
}
```

### Understanding the Results

```sep
pattern result_interpretation {
    signal_data = "market_data"
    
    coherence = measure_coherence(signal_data)
    entropy = measure_entropy(signal_data)
    
    // Interpret coherence
    if coherence > 0.8 {
        coherence_level = "Highly coherent - strong patterns"
    } else if coherence > 0.5 {
        coherence_level = "Moderately coherent - some patterns"
    } else {
        coherence_level = "Low coherence - chaotic or noisy"
    }
    
    // Interpret entropy
    if entropy < 0.2 {
        entropy_level = "Very predictable"
    } else if entropy < 0.5 {
        entropy_level = "Somewhat predictable"
    } else {
        entropy_level = "Highly unpredictable"
    }
    
    // Combined analysis
    signal_quality = coherence * (1.0 - entropy)
    
    print("Signal Analysis:")
    print("  Coherence:", coherence, "-", coherence_level)
    print("  Entropy:", entropy, "-", entropy_level)
    print("  Overall Quality:", signal_quality)
}
```

## Examples Walkthrough

### Example 1: Sensor Monitoring

```sep
pattern temperature_monitor {
    // Simulate temperature sensor data
    temp_data = "temperature_sensor_001"
    
    // Analyze the signal
    temp_coherence = measure_coherence(temp_data)
    temp_entropy = measure_entropy(temp_data)
    
    // Calculate health score
    sensor_health = temp_coherence * (1.0 - temp_entropy)
    
    // Status determination
    if sensor_health > 0.8 {
        status = "GOOD"
        maintenance_needed = false
    } else if sensor_health > 0.5 {
        status = "WARNING"
        maintenance_needed = false
    } else {
        status = "CRITICAL"
        maintenance_needed = true
    }
    
    print("Temperature Sensor Status:")
    print("  Health Score:", sensor_health)
    print("  Status:", status)
    print("  Maintenance Needed:", maintenance_needed)
}
```

### Example 2: Financial Analysis

```sep
pattern forex_signal {
    // Market data for EUR/USD
    eur_usd_data = "EUR_USD_M15"
    
    // Multi-timeframe analysis
    coherence_15m = measure_coherence(eur_usd_data)
    entropy_15m = measure_entropy(eur_usd_data)
    
    // Signal generation
    is_coherent = coherence_15m > 0.6
    is_stable = entropy_15m < 0.4
    
    // Trading decision
    buy_signal = is_coherent && is_stable
    signal_strength = coherence_15m * (1.0 - entropy_15m)
    
    print("Forex Analysis:")
    print("  Pair: EUR/USD")
    print("  Timeframe: 15M")
    print("  Coherence:", coherence_15m)
    print("  Stability:", is_stable)
    print("  Buy Signal:", buy_signal)
    print("  Signal Strength:", signal_strength)
    
    if buy_signal {
        print("🚀 RECOMMENDATION: Consider LONG position")
    } else {
        print("⏳ RECOMMENDATION: Wait for better signal")
    }
}
```

### Example 3: Pattern Recognition

```sep
pattern image_analysis {
    // Image or signal data
    image_data = "camera_feed_frame"
    
    // Extract patterns
    bit_representation = extract_bits(image_data)
    
    // Analyze patterns
    pattern_coherence = measure_coherence(image_data)
    pattern_complexity = measure_entropy(image_data)
    
    // Quantum field analysis
    qfh_signature = qfh_analyze(bit_representation)
    
    // Feature detection
    has_structure = pattern_coherence > 0.7
    is_complex = pattern_complexity > 0.5
    
    // Classification
    if has_structure && !is_complex {
        classification = "Simple Pattern"
    } else if has_structure && is_complex {
        classification = "Complex Pattern"
    } else {
        classification = "No Clear Pattern"
    }
    
    print("Image Analysis Results:")
    print("  Classification:", classification)
    print("  Structure Quality:", pattern_coherence)
    print("  Complexity:", pattern_complexity)
    print("  QFH Signature:", qfh_signature)
}
```

## Language Bindings

### Ruby Integration

```ruby
require 'sep_dsl'

# Create interpreter
interp = SEP::Interpreter.new

# Execute SEP code
interp.execute(<<~SCRIPT)
  pattern analysis {
    data = "sample_input"
    coherence = measure_coherence(data)
    entropy = measure_entropy(data)
    quality = coherence * (1.0 - entropy)
  }
SCRIPT

# Access results
coherence = interp['analysis.coherence']
quality = interp['analysis.quality']

puts "Coherence: #{coherence}"
puts "Quality: #{quality}"

# Quick analysis method
results = SEP.analyze("market_data")
puts "Quick analysis: #{results[:coherence]}"
```

### Python Integration (Coming Soon)

```python
import sep_dsl

# Quick pattern analysis
results = sep_dsl.analyze("sensor_data")
print(f"Coherence: {results['coherence']}")
print(f"Entropy: {results['entropy']}")

# Full interpreter
interp = sep_dsl.Interpreter()
interp.execute("""
  pattern python_demo {
    value = measure_coherence("data")
    print("Result:", value)
  }
""")
```

## Performance Tips

### 1. Use Appropriate Data Sizes

```sep
pattern performance_demo {
    // Optimal window sizes for different analyses
    short_term = "data_256_samples"    // Good for real-time
    medium_term = "data_1024_samples"  // Balanced accuracy/speed
    long_term = "data_4096_samples"    // High accuracy
    
    // Choose based on your needs
    coherence = measure_coherence(medium_term)
}
```

### 2. Cache Expensive Computations

```sep
pattern caching_demo {
    data_stream = "live_feed"
    
    // Compute once, use multiple times
    coherence_value = measure_coherence(data_stream)
    entropy_value = measure_entropy(data_stream)
    
    // Reuse results
    quality_1 = coherence_value * (1.0 - entropy_value)
    quality_2 = coherence_value * 0.8
    threshold_check = coherence_value > 0.6
}
```

### 3. GPU Acceleration

When CUDA is available, SEP DSL automatically uses GPU acceleration for:
- Coherence calculations
- Entropy analysis
- Pattern recognition
- Quantum field harmonics

No code changes needed - the engine automatically optimizes based on available hardware.

## Debugging and Troubleshooting

### Common Issues

1. **"File not found" errors**
   ```bash
   # Make sure the interpreter path is correct
   which sep_dsl_interpreter
   
   # Or use absolute path
   /full/path/to/build/src/dsl/sep_dsl_interpreter yourfile.sep
   ```

2. **Syntax errors**
   ```sep
   pattern debug_example {
       // Make sure statements end properly
       x = 42      // Good
       y = 3.14    // Good
       
       // Check parentheses and quotes
       result = measure_coherence("data")  // Good
       bad_result = measure_coherence(data  // Missing closing parenthesis
   }
   ```

3. **Variable access issues**
   ```sep
   pattern access_demo {
       internal_var = 100
   }
   
   pattern consumer {
       // Use dot notation for pattern member access
       value = access_demo.internal_var  // Correct
       // Not: value = internal_var      // Wrong - not in scope
   }
   ```

### Debug Output

Add print statements to understand what's happening:

```sep
pattern debug_trace {
    data = "test_input"
    
    print("Starting analysis of:", data)
    
    coherence = measure_coherence(data)
    print("Coherence calculated:", coherence)
    
    entropy = measure_entropy(data)
    print("Entropy calculated:", entropy)
    
    result = coherence * (1.0 - entropy)
    print("Final result:", result)
}
```

## Next Steps

### Beginner Path
1. ✅ Complete this getting started guide
2. 📚 Work through `examples/beginner/` 
3. 🎯 Try the sensor monitoring example
4. 🔧 Experiment with different AGI functions

### Intermediate Path
1. 📈 Study the forex trading example
2. 🔗 Learn pattern composition and member access
3. 🎨 Try language bindings (Ruby)
4. ⚡ Optimize for performance

### Advanced Path
1. 🛠️ Contribute to the project (see CONTRIBUTING.md)
2. 🧩 Build custom analysis functions
3. 🌐 Create language bindings for other languages
4. 🚀 Deploy in production environments

### Resources

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory with progressively complex samples
- **Contributing**: `CONTRIBUTING.md` for development guidelines
- **Issues**: GitHub issues for questions and bug reports
- **Community**: GitHub Discussions for general questions

Welcome to the SEP DSL community! 🚀
