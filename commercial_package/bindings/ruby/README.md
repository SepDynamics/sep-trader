# SEP DSL Ruby Bindings

Ruby bindings for the SEP (Sourcegraph Engine Platform) DSL - an AGI Coherence Framework with CUDA acceleration for quantum pattern analysis.

## Installation

```bash
gem install sep_dsl
```

## Quick Start

```ruby
require 'sep_dsl'

# Create an interpreter
interp = SEP::Interpreter.new

# Execute DSL code
interp.execute(<<~SCRIPT)
  pattern test {
    x = 10
    y = 5
    result = x + y
    coherence = measure_coherence("sensor_data")
    entropy = measure_entropy("sensor_data")
  }
SCRIPT

# Get results
puts "Result: #{interp['test.result']}"
puts "Coherence: #{interp['test.coherence']}"
puts "Entropy: #{interp['test.entropy']}"
```

## Quantum Pattern Analysis

```ruby
# Quick analysis
results = SEP.analyze("forex_data_sample")
puts "Coherence: #{results[:coherence]}"
puts "Entropy: #{results[:entropy]}"

# Detailed analysis
interp = SEP::Interpreter.new
interp.run_pattern("forex_analysis") do
  <<~DSL
    data = "EUR_USD_M1_sample"
    bits = extract_bits(data)
    coherence = measure_coherence(data)
    entropy = measure_entropy(data)
    
    is_coherent = coherence > 0.5
    is_stable = entropy < 0.5
    trade_signal = is_coherent && is_stable
  DSL
end

puts "Trade Signal: #{interp['forex_analysis.trade_signal']}"
```

## File Execution

```ruby
# Execute a .sep file
interp = SEP::Interpreter.new
interp.run_file("my_pattern.sep")

# Or using convenience method
interp.execute_file("my_pattern.sep")
```

## System Information

```ruby
puts "SEP Version: #{SEP.version}"
puts "CUDA Support: #{SEP.has_cuda?}"
```

## Requirements

- SEP DSL Engine (libsep.so) must be installed
- CUDA support is optional but recommended for performance
- Ruby 2.5 or higher

## License

MIT License
