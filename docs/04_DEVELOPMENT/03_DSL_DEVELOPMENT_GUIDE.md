# SEP DSL Development Guide

## Architecture Overview

SEP DSL is a comprehensive AGI pattern analysis platform with multiple components:

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DSL Engine    │    │  Language SDKs  │    │   Developer     │
│                 │    │                 │    │     Tools       │
│ • Parser/AST    │◄──►│ • Python        │◄──►│ • LSP Server    │
│ • Interpreter   │    │ • JavaScript    │    │ • VSCode Ext    │
│ • C++ Core      │    │ • Ruby          │    │ • Syntax HL     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CUDA Engine   │    │   API Server    │    │ Package Mgmt    │
│                 │    │                 │    │                 │
│ • QFH Analysis  │    │ • REST API      │    │ • npm           │
│ • Quantum Math  │    │ • WebSocket     │    │ • PyPI          │
│ • GPU Accel     │    │ • Rate Limiting │    │ • RubyGems      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Development Environment Setup

### Prerequisites

```bash
# Required tools
- C++17 compiler (clang++ or g++)
- CMake 3.16+
- CUDA Toolkit 12.9 (optional, for GPU acceleration)
- Node.js 14+ (for JavaScript SDK and tools)
- Python 3.8+ (for Python SDK)
- Ruby 2.7+ (for Ruby SDK)
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/SepDynamics/sep-dsl.git
cd sep-dsl

# Build core engine
./build.sh

# Test core functionality
./build/src/dsl/sep_dsl_interpreter examples/agi_demo_simple.sep

# Build all packages
./scripts/build_packages.sh
```

## Component Development

### 1. Core DSL Engine (`/src/`)

The heart of SEP DSL written in C++:

```cpp
// Key files:
src/dsl/ast/          // Abstract Syntax Tree nodes
src/dsl/lexer/        // Tokenization
src/dsl/parser/       // DSL parsing
src/dsl/runtime/      // Execution engine
src/core/             // C++ to CUDA bridge
```

**Development workflow:**
```bash
# Modify core engine
vim src/dsl/runtime/interpreter.cpp

# Build and test
./build.sh
./build/tests/dsl_interpreter_test
```

### 2. Language Bindings (`/bindings/`)

Multi-language support through C API:

#### Python SDK (`/bindings/python/`)
```bash
cd bindings/python
python setup.py build_ext --inplace
python -c "import sep_dsl; print('OK')"
```

#### JavaScript SDK (`/bindings/javascript/`)
```bash
cd bindings/javascript  
npm install
npm run build
node -e "const {DSLInterpreter} = require('./lib'); console.log('OK')"
```

#### Ruby SDK (`/bindings/ruby/`)
```bash
cd bindings/ruby
ruby -Ilib -rsep_dsl -e "puts 'OK'"
```

### 3. Developer Tools (`/tools/`)

#### Language Server (`/tools/lsp/`)
```bash
cd tools/lsp
npm install && npm run build
./bin/sep-dsl-language-server --stdio
```

#### VSCode Extension (`/tools/vscode-extension/`)
```bash
cd tools/vscode-extension
npm install && npm run compile
# Test in VSCode: F5 to launch Extension Development Host
```

### 4. API Server (`/api/`)

REST and WebSocket API:
```bash
cd api
npm install && npm run build
npm start  # Runs on http://localhost:3000
```

## Testing Strategy

### Unit Tests
```bash
# C++ core tests
./build/tests/dsl_parser_test
./build/tests/dsl_interpreter_test

# Python tests  
cd bindings/python && python -m pytest tests/

# JavaScript tests
cd bindings/javascript && npm test

# Ruby tests
cd bindings/ruby && ruby test/test_dsl.rb
```

### Integration Tests
```bash
# End-to-end DSL execution
./build/src/dsl/sep_dsl_interpreter examples/agi_demo_simple.sep

# API server tests
cd api && npm test

# Language server tests
cd tools/lsp && npm test
```

### Performance Tests
```bash
# CUDA performance profiling
nsys profile ./build/examples/pattern_metric_example Testing/OANDA/

# Memory leak detection
valgrind --leak-check=full ./build/src/dsl/sep_dsl_interpreter examples/test.sep
```

## Code Style Guidelines

### C++
- Follow existing patterns in `src/`
- Use `snake_case` for functions/variables
- Use `PascalCase` for classes
- RAII for resource management
- Prefer `std::unique_ptr` over raw pointers

### JavaScript/TypeScript
- ES6+ features
- Async/await for promises
- JSDoc comments for public APIs
- Prettier for formatting

### Python
- PEP 8 compliance
- Type hints for public APIs
- Docstrings for classes/functions
- Black for formatting

## Release Process

### Version Bumping
```bash
# Update version in all package.json, setup.py, gemspec files
# Update CHANGELOG.md
# Tag release
git tag v1.0.0
git push origin v1.0.0
```

### Package Publishing
```bash
# Build all packages
./scripts/build_packages.sh

# Publish to registries
cd bindings/python && python -m twine upload dist/*
cd bindings/javascript && npm publish
cd bindings/ruby && gem push sep_dsl-*.gem
cd tools/lsp && npm publish
```

## Debugging Tips

### DSL Execution Issues
```bash
# Enable verbose logging
DEBUG=1 ./build/src/dsl/sep_dsl_interpreter script.sep

# Check AST generation
./build/src/dsl/sep_dsl_interpreter --ast-dump script.sep
```

### C++ Core Issues
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# GDB debugging
gdb ./build/src/dsl/sep_dsl_interpreter
(gdb) run examples/test.sep
```

### Language Binding Issues
```bash
# Python extension debugging
cd bindings/python
python setup.py build_ext --inplace --debug

# Node.js native debugging
cd bindings/javascript
node-gyp rebuild --debug
```

## Performance Optimization

### Profiling Tools
```bash
# CPU profiling
perf record ./build/src/dsl/sep_dsl_interpreter script.sep
perf report

# CUDA profiling
nsys profile --output=profile ./build/examples/pattern_metric_example

# Memory profiling
valgrind --tool=massif ./build/src/dsl/sep_dsl_interpreter script.sep
```

### Optimization Targets
- **Parser**: Reduce allocations in AST construction
- **Interpreter**: Implement bytecode compilation
- **Engine**: Optimize CUDA kernel launches
- **Bindings**: Minimize C++ ↔ Language marshaling

## Contributing Workflow

### Setting Up Development Branch
```bash
git checkout -b feature/your-feature-name
# Make changes
git add .
git commit -m "feat: description of changes"
git push origin feature/your-feature-name
# Create pull request
```

### Code Review Checklist
- [ ] Tests pass (`./build.sh && npm test`)
- [ ] Documentation updated
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Backward compatibility maintained

## Advanced Development

### Adding New Built-in Functions

1. **Define in engine** (`src/core/facade.cpp`):
```cpp
core::Result<double> measure_stability(const std::string& data_name) {
    // Implementation
    return core::Result<double>::success(stability_value);
}
```

2. **Register in interpreter** (`src/dsl/runtime/interpreter.cpp`):
```cpp
builtin_functions_["measure_stability"] = [this](const std::vector<Value>& args) -> Value {
    // Validation and engine call
};
```

3. **Update language bindings** documentation and tests

### Extending Language Support

1. **Create binding directory**: `bindings/new-language/`
2. **Implement C API wrapper**: Use existing C API in `src/io/sep_c_api.h`
3. **Add to build script**: Update `scripts/build_packages.sh`
4. **Update documentation**: Add to main README

### Performance Monitoring

```bash
# Runtime metrics collection
export SEP_METRICS=1
./build/src/dsl/sep_dsl_interpreter script.sep

# API server metrics
curl http://localhost:3000/metrics  # Prometheus format
```
