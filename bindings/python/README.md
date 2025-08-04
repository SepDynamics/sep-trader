# SEP DSL Python SDK

Advanced AGI Pattern Analysis Language for Python

## Overview

SEP DSL is a domain-specific language for advanced AGI pattern analysis, providing quantum coherence analysis, entropy measurement, and sophisticated pattern recognition capabilities through an intuitive DSL syntax.

## Features

- **Real-time Quantum Analysis**: Coherence and entropy measurement
- **CUDA Acceleration**: GPU-powered pattern recognition
- **Advanced AGI Algorithms**: Quantum field harmonics and trajectory damping
- **Production-Grade**: Mathematical validation and testing
- **Pythonic Interface**: Clean, easy-to-use Python wrapper

## Installation

### Prerequisites

- Python 3.8+
- SEP DSL core library (`libsep.so`)
- C compiler for building extensions

### From Source

```bash
# Clone the repository
git clone https://github.com/SepDynamics/sep-dsl.git
cd sep-dsl/bindings/python

# Build and install
python setup.py build_ext --inplace
python setup.py install
```

### Using pip (when available)

```bash
# Build from source - not yet published
```

## Quick Start

```python
import sep_dsl

# Create DSL interpreter
dsl = sep_dsl.DSLInterpreter()

# Execute DSL script
dsl.execute('''
    pattern sensor_analysis {
        coherence = measure_coherence("sensor_data")
        entropy = measure_entropy("sensor_data") 
        print("Coherence:", coherence, "Entropy:", entropy)
    }
''')

# Get results
coherence = dsl.get_variable("sensor_analysis.coherence")
entropy = dsl.get_variable("sensor_analysis.entropy")

print(f"Analysis: Coherence={coherence}, Entropy={entropy}")
```

## API Reference

### DSLInterpreter

Main interface to the SEP DSL engine.

#### Methods

- **`execute(script: str)`** - Execute DSL script
- **`execute_file(filepath: str)`** - Execute DSL script from file
- **`get_variable(name: str) -> str`** - Get variable value (supports dot notation)
- **`get_pattern_results(pattern_name: str) -> Dict[str, str]`** - Get all pattern variables
- **`analyze_coherence(data_name: str) -> float`** - Quick coherence analysis
- **`analyze_entropy(data_name: str) -> float`** - Quick entropy analysis

### Convenience Functions

- **`quick_analysis(data_name: str) -> Dict[str, float]`** - Fast coherence and entropy analysis

## Examples

### Basic Pattern Analysis

```python
import sep_dsl

dsl = sep_dsl.DSLInterpreter()

# Complex pattern with multiple metrics
dsl.execute('''
    pattern comprehensive_analysis {
        coherence = measure_coherence("data_stream")
        entropy = measure_entropy("data_stream")
        bits = extract_bits("data_stream")
        rupture = qfh_analyze(bits)
        
        if (coherence > 0.8 && entropy < 0.3) {
            print("High-quality signal detected")
        }
    }
''')

# Get all results
results = dsl.get_pattern_results("comprehensive_analysis")
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### File-based Execution

```python
# Save DSL script to file
with open("analysis.sep", "w") as f:
    f.write('''
        pattern file_analysis {
            coherence = measure_coherence("sensor_input")
            stability = measure_entropy("sensor_input")
            print("File-based analysis complete")
        }
    ''')

# Execute from file
dsl = sep_dsl.DSLInterpreter()
dsl.execute_file("analysis.sep")

coherence = float(dsl.get_variable("file_analysis.coherence"))
```

### Quick Analysis Helper

```python
# One-liner analysis
results = sep_dsl.quick_analysis("my_data")
print(f"Coherence: {results['coherence']:.3f}")
print(f"Entropy: {results['entropy']:.3f}")
```

### Real-time Data Processing

```python
import sep_dsl
import time

dsl = sep_dsl.DSLInterpreter()

def process_sensor_data(data_stream):
    # Quick coherence check
    coherence = dsl.analyze_coherence("live_stream")
    
    if coherence > 0.7:
        # High coherence - do full analysis
        dsl.execute('''
            pattern detailed_analysis {
                entropy = measure_entropy("live_stream")
                bits = extract_bits("live_stream") 
                rupture = qfh_analyze(bits)
                optimized = manifold_optimize("pattern", 0.8, 0.9)
            }
        ''')
        
        results = dsl.get_pattern_results("detailed_analysis")
        return results
    
    return {"coherence": coherence, "status": "low_quality"}

# Simulate real-time processing
for i in range(10):
    results = process_sensor_data(f"stream_{i}")
    print(f"Tick {i}: {results}")
    time.sleep(1)
```

## Error Handling

```python
import sep_dsl

try:
    dsl = sep_dsl.DSLInterpreter()
    dsl.execute("invalid syntax here")
except sep_dsl.DSLRuntimeError as e:
    print(f"Script execution failed: {e}")

try:
    value = dsl.get_variable("nonexistent.variable")
except sep_dsl.DSLVariableError as e:
    print(f"Variable not found: {e}")
```

## Testing

```bash
# Run test suite
cd /sep/bindings/python
python -m pytest tests/ -v

# Or use unittest
python tests/test_dsl.py
```

## Integration with Scientific Python

### NumPy Integration

```python
import sep_dsl
import numpy as np

# Analyze NumPy arrays (conceptual - requires data ingestion)
data = np.random.random(1000)

dsl = sep_dsl.DSLInterpreter()
# In real implementation, you'd feed data to the engine first
results = sep_dsl.quick_analysis("numpy_data")

print(f"Array coherence: {results['coherence']}")
print(f"Array entropy: {results['entropy']}")
```

### Pandas Integration

```python
import sep_dsl
import pandas as pd

# Process time series data
df = pd.read_csv("sensor_data.csv")

dsl = sep_dsl.DSLInterpreter()

# Analyze each column
analysis_results = []
for column in df.columns:
    results = sep_dsl.quick_analysis(column)
    results['column'] = column
    analysis_results.append(results)

# Create results DataFrame
results_df = pd.DataFrame(analysis_results)
print(results_df)
```

## Performance Tips

1. **Reuse Interpreters**: Create one `DSLInterpreter` instance and reuse it
2. **Batch Processing**: Group multiple patterns in single script execution
3. **Quick Methods**: Use `analyze_coherence()` and `analyze_entropy()` for simple cases
4. **Pattern Caching**: Results are cached within the same interpreter session

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Support

- Issues: https://github.com/SepDynamics/sep-dsl/issues
- Documentation: https://github.com/SepDynamics/sep-dsl#readme
- Email: contact@example.com
