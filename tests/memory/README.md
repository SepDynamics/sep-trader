# SEP DSL Memory Safety Testing

This directory contains comprehensive memory leak detection and validation tools for the SEP DSL parser and interpreter, designed to ensure production-grade memory safety and prevent memory-related vulnerabilities.

## Overview

Memory safety testing uses **AddressSanitizer (ASAN)** and **Valgrind** to detect:
- Memory leaks
- Buffer overflows and underflows
- Use-after-free errors
- Double-free errors
- Memory corruption
- Stack overflow detection

## Quick Start

```bash
# Run complete memory testing suite
./run_memory_tests.sh

# Run only AddressSanitizer tests
./run_memory_tests.sh asan

# Run only Valgrind tests  
./run_memory_tests.sh valgrind
```

## Testing Tools

### AddressSanitizer (ASAN)
- **Purpose**: Fast runtime memory error detection
- **Coverage**: Buffer overflows, use-after-free, memory leaks
- **Performance**: ~2x slowdown, suitable for continuous testing
- **Integration**: Built with clang compiler flags

### LeakSanitizer (LSAN)
- **Purpose**: Memory leak detection (part of ASAN)
- **Coverage**: Detects memory that's allocated but never freed
- **Reporting**: Detailed leak reports with stack traces

### Valgrind
- **Purpose**: Comprehensive memory debugging and profiling
- **Coverage**: All memory errors plus performance analysis
- **Performance**: ~10-50x slowdown, for thorough validation
- **Detail**: Extremely detailed error reports

## Build Requirements

### Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install valgrind clang build-essential

# Verify installation
valgrind --version
clang++ --version
clang++ --help | grep "fsanitize=address"  # Should show ASAN support
```

### Compiler Support
- **Clang**: Full AddressSanitizer and LeakSanitizer support
- **GCC**: AddressSanitizer support (may vary by version)
- **Valgrind**: Works with any compiler

## Manual Testing

### AddressSanitizer Build
```bash
# Build with memory sanitizers
mkdir build_asan
cd build_asan

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS="-fsanitize=address -fsanitize=leak -g -O1" \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fsanitizer=leak -g -O1" \
    -DENABLE_MEMORY_TESTING=ON \
    -DSEP_USE_CUDA=OFF

make -j$(nproc) sep_dsl_interpreter memory_test_runner
```

### Running Tests
```bash
# Set AddressSanitizer options
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:print_stats=1"
export LSAN_OPTIONS="print_suppressions=0"

# Test DSL interpreter
./build_asan/src/dsl/sep_dsl_interpreter examples/hello_world.sep

# Run comprehensive memory tests
./build_asan/tests/memory/memory_test_runner

# Valgrind testing
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --verbose \
    ./build/src/dsl/sep_dsl_interpreter examples/hello_world.sep
```

## Test Coverage

### Core Components Tested
1. **DSL Parser** - Syntax analysis and AST generation
2. **DSL Interpreter** - Runtime execution and variable management
3. **Function Definitions** - User-defined function memory management
4. **Exception Handling** - Try/catch/finally memory safety
5. **Pattern Execution** - Pattern variable scoping and cleanup

### Test Scenarios
- **Basic Operations**: Simple pattern execution
- **Complex Parsing**: Nested patterns and expressions
- **Function Calls**: User-defined function invocation
- **Exception Flows**: Error handling paths
- **Stress Testing**: 1000+ iterations for leak detection
- **Memory Pressure**: Large pattern generation and parsing

## Interpreting Results

### AddressSanitizer Output
```
# Clean run (good)
=================================================================
==12345==ASAN==Live==heap==objects==
==12345==ASAN==Statistics==
==12345==InternalSymbolizer::InternalSymbolizer
==12345==ASAN==exiting==

# Memory error (bad)
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
==12345==READ of size 4 at 0x... thread T0:
    #0 0x... in function_name file.cpp:line
```

### Valgrind Output
```
# Clean run (good)
==12345== HEAP SUMMARY:
==12345==     in use at exit: 0 bytes in 0 blocks
==12345==   total heap usage: 1,234 allocs, 1,234 frees, 567,890 bytes allocated
==12345== All heap blocks were freed -- no leaks are possible

# Memory leak (bad)  
==12345== LEAK SUMMARY:
==12345==    definitely lost: 64 bytes in 1 blocks
==12345==    indirectly lost: 0 bytes in 0 blocks
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Memory Safety Testing
  run: |
    sudo apt-get update
    sudo apt-get install valgrind clang
    ./run_memory_tests.sh
    
    # Fail if memory issues detected
    if grep -q "ERROR:" output/asan_test_log.txt; then
      echo "AddressSanitizer detected memory errors"
      exit 1
    fi
```

### Build Integration
```bash
# Add to build scripts
./build.sh && ./run_memory_tests.sh asan
```

## Troubleshooting

### Common Issues

1. **ASAN not available**
   ```bash
   # Solution: Install clang with ASAN support
   sudo apt install clang
   ```

2. **Valgrind not found**
   ```bash
   # Solution: Install valgrind
   sudo apt install valgrind
   ```

3. **False positives**
   ```bash
   # Solution: Use suppression files
   valgrind --suppressions=valgrind.supp ./program
   ```

4. **CUDA conflicts with ASAN**
   ```bash
   # Solution: Disable CUDA for memory testing
   cmake -DSEP_USE_CUDA=OFF
   ```

### Debugging Memory Issues

1. **Get detailed stack traces**
   ```bash
   export ASAN_OPTIONS="symbolize=1:print_stacktrace=1"
   ```

2. **Enable verbose output**
   ```bash
   valgrind --verbose --track-origins=yes
   ```

3. **Check specific allocation**
   ```bash
   valgrind --track-fds=yes --show-leak-kinds=definite
   ```

## Performance Impact

| Tool | Slowdown | Memory Usage | Use Case |
|------|----------|--------------|----------|
| AddressSanitizer | ~2x | ~2x | Continuous testing |
| LeakSanitizer | ~1.5x | ~1.5x | Automated CI/CD |
| Valgrind | ~10-50x | ~1x | Deep debugging |

## Expected Results

For a properly implemented DSL:
- **Zero memory leaks** detected by both ASAN and Valgrind
- **Zero buffer overflows** or use-after-free errors
- **Clean heap summary** showing all allocations properly freed
- **Stress tests pass** with 1000+ iterations without memory growth

This comprehensive memory testing ensures SEP DSL meets production-grade memory safety standards.
