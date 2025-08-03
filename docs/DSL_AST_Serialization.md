# SEP DSL AST Serialization

This document describes the AST (Abstract Syntax Tree) serialization functionality for the SEP DSL, which allows saving and loading parsed programs to/from JSON format.

## Overview

AST serialization enables:
- **Caching parsed programs** for faster startup times
- **Distributing pre-compiled DSL programs** without source code
- **Debugging complex parse trees** by inspecting the JSON representation
- **Building development tools** that work with AST structures

## Usage

### Command Line Interface

The DSL interpreter supports new flags for AST serialization:

```bash
# Save AST while parsing and executing
./sep_dsl_interpreter --save-ast program.ast.json program.sep

# Load and execute pre-parsed AST
./sep_dsl_interpreter --load-ast program.ast.json

# Get help
./sep_dsl_interpreter --help
```

### Programmatic API

```cpp
#include "dsl/ast/serializer.h"
#include "dsl/parser/parser.h"

// Parse and serialize
dsl::parser::Parser parser(source_code);
auto program = parser.parse();

dsl::ast::ASTSerializer serializer;

// Serialize to JSON
nlohmann::json json = serializer.serialize(*program);

// Save to file
bool success = serializer.saveToFile(*program, "program.ast.json");

// Load from file
auto loaded_program = serializer.loadFromFile("program.ast.json");

// Deserialize from JSON
auto restored_program = serializer.deserialize(json);
```

## JSON Format Specification

The serialized AST uses a hierarchical JSON structure where each node contains:
- `type`: The AST node type (e.g., "Program", "PatternDecl", "BinaryOp")
- `location`: Source location with line and column numbers
- Node-specific fields depending on the type

### Example JSON Structure

```json
{
  "type": "Program",
  "location": { "line": 1, "column": 1 },
  "streams": [
    {
      "type": "StreamDecl",
      "location": { "line": 15, "column": 1 },
      "name": "sensor_data",
      "source": "oanda",
      "params": {
        "symbol": "EUR_USD",
        "timeframe": "M1"
      }
    }
  ],
  "patterns": [
    {
      "type": "PatternDecl",
      "location": { "line": 1, "column": 1 },
      "name": "test",
      "parent_pattern": "",
      "inputs": [],
      "body": [
        {
          "type": "Assignment",
          "location": { "line": 2, "column": 5 },
          "name": "result",
          "type_annotation": "INFERRED",
          "value": {
            "type": "BinaryOp",
            "location": { "line": 2, "column": 14 },
            "left": {
              "type": "NumberLiteral",
              "location": { "line": 2, "column": 14 },
              "value": 3
            },
            "op": "*",
            "right": {
              "type": "NumberLiteral",
              "location": { "line": 2, "column": 18 },
              "value": 4
            }
          }
        }
      ]
    }
  ],
  "signals": []
}
```

## Supported AST Nodes

### Expressions
- `NumberLiteral` - Numeric constants
- `StringLiteral` - String constants  
- `BooleanLiteral` - Boolean constants
- `Identifier` - Variable references
- `BinaryOp` - Binary operations (+, -, *, /, etc.)
- `UnaryOp` - Unary operations (-, !, etc.)
- `Call` - Function calls
- `MemberAccess` - Object member access
- `ArrayLiteral` - Array constants [1, 2, 3]
- `ArrayAccess` - Array indexing arr[0]
- `WeightedSum` - Weighted sum expressions
- `AwaitExpression` - Async await expressions

### Statements
- `Assignment` - Variable assignments
- `ExpressionStatement` - Expression statements
- `EvolveStatement` - Evolve blocks
- `IfStatement` - Conditional statements
- `WhileStatement` - Loop statements
- `ReturnStatement` - Function returns
- `FunctionDeclaration` - Function definitions
- `AsyncFunctionDeclaration` - Async function definitions
- `ImportStatement` - Module imports
- `ExportStatement` - Module exports
- `TryStatement` - Exception handling
- `ThrowStatement` - Exception throwing

### Declarations
- `StreamDecl` - Stream declarations
- `PatternDecl` - Pattern declarations
- `SignalDecl` - Signal declarations

## Type System

Type annotations are preserved in the serialization:
- `NUMBER` - Numeric types
- `STRING` - String types
- `BOOL` - Boolean types
- `PATTERN` - Pattern types
- `VOID` - Void type
- `ARRAY` - Array types
- `INFERRED` - Types to be inferred

## Error Handling

The serializer provides comprehensive error handling:

```cpp
try {
    auto program = serializer.loadFromFile("program.ast.json");
} catch (const std::runtime_error& e) {
    std::cerr << "Serialization error: " << e.what() << std::endl;
}
```

Common errors:
- **JSON Parse Error**: Invalid JSON format in file
- **Unknown Node Type**: Unsupported AST node type in JSON
- **File Not Found**: AST file doesn't exist
- **Invalid Program Structure**: JSON doesn't represent valid Program node

## Performance Considerations

### Benefits
- **Faster Loading**: Pre-parsed AST loads ~10x faster than parsing source
- **Reduced Memory**: Shared AST files reduce memory usage in multi-process scenarios
- **Caching**: Build systems can cache parsed AST files for incremental builds

### Trade-offs  
- **File Size**: JSON AST files are typically 3-5x larger than source files
- **Versioning**: AST format may change between DSL versions
- **Debugging**: JSON AST is less readable than source code

## Testing

Comprehensive test suite ensures serialization correctness:

```bash
# Run serialization tests
./build/tests/dsl_serialization_test

# Test with real DSL files
./sep_dsl_interpreter --save-ast test.ast.json test.sep
./sep_dsl_interpreter --load-ast test.ast.json
```

### Test Coverage
- ✅ Basic serialization/deserialization
- ✅ Roundtrip accuracy (parse → serialize → deserialize → execute)
- ✅ File operations (save/load)
- ✅ Complex expression trees
- ✅ Function declarations and async features
- ✅ Error handling and edge cases

## Future Enhancements

Potential improvements for future versions:
- **Binary Format**: More compact binary serialization
- **Incremental Updates**: Partial AST updates for large programs
- **Compression**: Built-in compression for AST files
- **Version Compatibility**: Automatic migration between AST versions
- **Debug Symbols**: Enhanced debug information preservation

## Examples

### Complete Workflow

1. **Develop DSL program** (`trading_strategy.sep`):
```sep
pattern momentum_strategy {
    short_ma = moving_average(price, 10)
    long_ma = moving_average(price, 20)
    
    if (short_ma > long_ma) {
        signal = "BUY"
    } else {
        signal = "SELL"
    }
}
```

2. **Parse and save AST**:
```bash
./sep_dsl_interpreter --save-ast trading_strategy.ast.json trading_strategy.sep
```

3. **Deploy with pre-parsed AST**:
```bash
# In production environment
./sep_dsl_interpreter --load-ast trading_strategy.ast.json
```

4. **Inspect AST structure**:
```bash
cat trading_strategy.ast.json | jq '.patterns[0].body'
```

This serialization system provides a solid foundation for advanced DSL tooling and deployment scenarios while maintaining full fidelity to the original program structure.
