# SEP-DSL Development Branch

This branch contains the DSL foundation for the SEP quantum trading engine. The DSL provides a high-level, domain-specific language for expressing coherence computations while preserving access to all existing C++ engine functionality.

## Status: Weekend Exploration Ready ✅

The complete foundation is implemented and building successfully. All core infrastructure is in place for development.

## Quick Start

```bash
# Build the DSL foundation
./build.sh

# Test the foundation
./build/examples/dsl_test

# Explore the example DSL syntax
cat docs/dsl/examples/forex_coherence.sep
```

## Architecture Overview

```
DSL Script → Parser → AST → Compiler → EngineOperations → SEP Engine
```

### Core Components

1. **AST (Abstract Syntax Tree)**: `src/dsl/ast/nodes.h`
   - Defines all DSL language constructs
   - Pattern definitions, expressions, signals, memory rules

2. **Parser**: `src/dsl/parser/`
   - Converts DSL text into AST structures
   - **Weekend Focus**: Implement `parsePattern()` method

3. **Compiler**: `src/dsl/compiler/`
   - Translates AST into executable operations
   - **Weekend Focus**: Map DSL operations to engine calls

4. **Runtime**: `src/dsl/runtime/`
   - Orchestrates parsing, compilation, and execution
   - Integrates with existing EngineFacade and MemoryTierManager

### DSL Syntax Example

```sep
// Define a coherence computation pattern
pattern forex_coherence {
    input: market_data
    
    // Apply quantum analysis
    qfh_result = qfh(bits: extract_bits(input))
    qbsa_result = qbsa(pattern: input)
    
    // Compute weighted coherence
    coherence = weighted_sum {
        qfh_result.coherence: 0.3
        qbsa_result.stability: 0.7
    }
}

// Define trading signal
signal buy_signal {
    trigger: forex_coherence.coherence > 0.85
    action: BUY
}
```

## Weekend Development Plan

### Priority 1: Basic Parser Implementation

**File**: `src/dsl/parser/parser.cpp`

**Goal**: Parse simple pattern definitions into AST

**Key Methods to Implement**:
- `parsePattern()` - Parse pattern blocks
- `parseExpression()` - Parse arithmetic/logical expressions  
- `parseFunctionCall()` - Parse operations like qfh(), qbsa()

**Test**: Parse the forex_coherence.sep example successfully

### Priority 2: Core Compiler Mapping

**File**: `src/dsl/compiler/compiler.cpp`

**Goal**: Map DSL operations to existing SEP engine calls

**Key Mappings to Implement**:
- `qfh()` → `QFHBasedProcessor::analyze()`
- `qbsa()` → `QBSAProcessor::analyze()`
- `manifold()` → `QuantumManifoldOptimizer::optimize()`
- `coherence()` → `QuantumProcessor::calculateCoherence()`

**Reference**: Your existing engine calls in `/src/quantum/` and `/src/engine/facade/`

### Priority 3: Simple End-to-End Test

**Goal**: Execute a basic pattern that calls into the real engine

**Steps**:
1. Write minimal DSL script
2. Parse to AST
3. Compile to operations  
4. Execute against EngineFacade
5. Verify actual engine calls are made

## Existing Engine Integration Points

### Core Primitives (from TASK.md analysis)

**Pattern Operations**:
- `DataParser::candlesToPatterns()` - Create patterns from data
- `PatternEvolutionBridge` - Pattern evolution
- `QuantumProcessor::calculateCoherence/Stability()` - Metrics

**Quantum Operations**:
- `QFHBasedProcessor::analyze()` - QFH analysis
- `QBSAProcessor::analyze()` - QBSA analysis  
- `QuantumManifoldOptimizer::optimize()` - Manifold optimization

**Memory Operations**:
- `MemoryTierManager::allocate()` - Store patterns
- `MemoryTierManager::promoteBlock()` - Tier promotion
- Pattern retrieval and querying

## Build System

The DSL is fully integrated into the existing build:

- **Library**: `libsep_dsl.a` 
- **Dependencies**: Links to `sep_engine`, `sep_quantum`, `sep_memory`
- **Test Executable**: `./build/examples/dsl_test`

## Files Structure

```
src/dsl/
├── ast/
│   └── nodes.h              # AST definitions ✅
├── parser/
│   ├── parser.h            # Parser interface ✅  
│   └── parser.cpp          # Parser implementation (TODO)
├── compiler/
│   ├── compiler.h          # Compiler interface ✅
│   └── compiler.cpp        # Compiler implementation (TODO)
├── runtime/
│   ├── runtime.h           # Runtime interface ✅
│   └── runtime.cpp         # Runtime implementation ✅
└── CMakeLists.txt          # Build configuration ✅

docs/dsl/
├── examples/
│   └── forex_coherence.sep # Target DSL syntax ✅
└── README.md               # This file ✅

examples/
└── dsl_test.cpp            # Foundation test ✅
```

## Development Guidelines

1. **Preserve Engine**: All existing functionality remains unchanged
2. **Gradual Implementation**: Start with basic patterns, expand incrementally  
3. **Real Integration**: DSL should make actual calls to your engine, not mocks
4. **Error Handling**: Comprehensive error reporting for invalid DSL
5. **Performance**: DSL compilation should be fast for interactive development

## Next Steps Beyond Weekend

1. **Advanced Parsing**: Full language feature support
2. **IDE Integration**: Syntax highlighting, autocompletion
3. **JIT Compilation**: Performance optimization for hot paths
4. **Standard Library**: Reusable pattern definitions
5. **Package System**: Modular DSL components

## Branch Management

- **Current Branch**: `dsl-development`
- **Original Branch**: `master` (preserved unchanged)
- **Merge Strategy**: Keep branches separate until DSL is production-ready

The foundation is ready. Happy weekend development! 🚀
