# SEP DSL Language Reference

**Version 1.2.0**

This document provides a complete reference for the SEP Domain-Specific Language (DSL), a specialized language for quantum-inspired pattern analysis.

## 1. Introduction

The SEP DSL is designed for expressing complex pattern recognition, signal generation, and decision-making processes. It provides a high-level, declarative interface to the powerful C++/CUDA engine, abstracting away the underlying complexity.

**Design Goals:**
- **Quantum-native:** Direct integration with Quantum Field Harmonics (QFH) and coherence analysis.
- **Performance:** CUDA-accelerated execution with minimal overhead.
- **Expressiveness:** Rich pattern matching and data transformation capabilities.
- **Safety:** Strong type hints, exception handling, and memory safety.

## 2. Quick Start Tutorial

This tutorial walks through creating a simple pattern analysis program.

### Step 1: Define a Data Stream
This tells the DSL where to get its data.
```sep
stream sensor_data from "temperature_readings.csv"
```

### Step 2: Define a Pattern
This is where you define the analysis logic.
```sep
pattern temperature_spike {
    input: sensor_data
    
    high_temp = measure_coherence(sensor_data) > 0.8
    rapid_change = qfh_analyze(sensor_data) > 0.7
}
```

### Step 3: Create a Signal
This defines an action to be taken when a pattern is detected.
```sep
signal alert_system {
    trigger: temperature_spike.high_temp
    confidence: temperature_spike.rapid_change
    action: SEND_ALERT
}
```

### Running the Program
Save the code as `monitor.sep` and run it:
```bash
./build/src/dsl/sep_dsl_interpreter monitor.sep
```

## 3. Language Concepts

### 3.1. Lexical Elements
- **Keywords:** `pattern`, `stream`, `signal`, `if`, `else`, `function`, etc.
- **Identifiers:** Start with a letter or underscore, case-sensitive.
- **Literals:** Numbers (`42`, `3.14`), Strings (`"hello"`), Booleans (`true`, `false`), and Arrays (`[1, 2, 3]`).

### 3.2. Types
- **`number`**: Integer or floating-point values.
- **`string`**: UTF-8 text data.
- **`bool`**: `true` or `false`.
- **`array`**: Ordered, mixed-type collections (e.g., `[1, "text", true]`).
- Type annotations are optional but recommended: `variable: number = 10`

### 3.3. Declarations

**Streams:** Define data sources.
```sep
stream forex_data from "oanda://EUR_USD"
```

**Patterns:** The core analysis blocks.
```sep
pattern coherence_spike {
    input: forex_data
    rupture_detected = qfh_analyze(forex_data) > 0.8
}
```

**Signals:** Define event-driven actions.
```sep
signal buy_signal {
    trigger: coherence_spike.rupture_detected
    action: BUY
}
```

### 3.4. Expressions & Statements
The language supports standard arithmetic (`+`, `-`, `*`, `/`), comparison (`>`, `==`, `!=`), and logical (`&&`, `||`, `!`) operators with conventional precedence rules. It also supports `if/else` statements and `while` loops.

### 3.5. Functions
User-defined functions are supported, including `async` functions for non-blocking operations.
```sep
async function fetch_and_analyze(sensor_id: string): number {
    raw_data = await fetch_sensor_data(sensor_id)
    entropy = await measure_entropy(raw_data)
    return entropy
}
```

## 4. Standard Library

The DSL includes a rich standard library, organized into modules.

- **Location:** `src/dsl/stdlib/`
- **Architecture:** The library is modular, with each category (math, statistical, etc.) in its own subdirectory. The central `stdlib.cpp` registers all functions with the DSL runtime.

### 4.1. Quantum Analysis Functions
- `measure_coherence(data)`: Measures quantum coherence.
- `measure_entropy(data)`: Calculates Shannon entropy.
- `qfh_analyze(data)`: Performs Quantum Field Harmonics analysis.

### 4.2. Other Libraries
- **Mathematical:** `sin`, `cos`, `sqrt`, `log`, etc.
- **Statistical:** `mean`, `median`, `stddev`, `correlation`.
- **Time Series:** `moving_average`, `trend_detection`.
- **Data Transformation:** `normalize`, `standardize`, `filter_above`.

## 5. Advanced Features

### 5.1. AST Serialization
The DSL Abstract Syntax Tree (AST) can be serialized to and from JSON. This is useful for caching parsed programs, debugging, and building developer tools.

**Usage:**
```bash
# Save the AST of a program
./sep_dsl_interpreter --save-ast program.ast.json program.sep

# Load and execute a pre-parsed AST
./sep_dsl_interpreter --load-ast program.ast.json
```

### 5.2. Exception Handling
The language supports `try...catch...finally` blocks for robust error handling.
```sep
pattern robust_analysis {
    try {
        data = measure_entropy("sensor_001")
        if (data > 0.9) { throw "Critical entropy!" }
    } catch (error) {
        print("Error caught:", error)
    } finally {
        print("Analysis complete.")
    }
}
```

## 6. Grammar Specification (BNF)

This is a simplified version of the grammar.

```bnf
Program         ::= Declaration*
Declaration     ::= StreamDecl | PatternDecl | SignalDecl
StreamDecl      ::= "stream" IDENTIFIER "from" STRING
PatternDecl     ::= "pattern" IDENTIFIER "{" Statement* "}"
SignalDecl      ::= "signal" IDENTIFIER "{" "trigger:" Expression "}"
Statement       ::= Assignment | ExpressionStatement | IfStatement
Expression      ::= LogicalOr
LogicalOr       ::= LogicalAnd ("||" LogicalAnd)*
...
```

## 7. Comprehensive Example

This example demonstrates using the DSL for multi-domain analysis.

```sep
// Financial Market Analysis
stream market_data from "financial/EUR_USD_M5.csv"

pattern market_volatility {
    input: market_data
    
    price_entropy = measure_entropy(market_data) > 0.65
    trend_rupture = qfh_analyze(market_data) > 0.75
    volatile_market = price_entropy && trend_rupture
}

signal trading_alert {
    trigger: market_volatility.volatile_market
    action: TRADE_ALERT
}

// Scientific Experiment Monitoring  
stream experiment_readings from "lab/quantum_measurement.json"

pattern quantum_decoherence {
    input: experiment_readings
    
    coherence_loss = measure_coherence(experiment_readings) < 0.3
    entropy_spike = measure_entropy(experiment_readings) > 0.8
    decoherence_event = coherence_loss && entropy_spike
}

signal experiment_flag {
    trigger: quantum_decoherence.decoherence_event
    action: RECALIBRATE
}
```
