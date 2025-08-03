# SEP DSL Regression Test Suite

## Overview

The regression test suite ensures backward compatibility across SEP DSL versions. These tests verify that core language features continue to work correctly as new features are added.

## Quick Start

```bash
# Run all regression tests
./tests/regression/run_regression_tests.sh

# Or run manually after build
./build.sh
./tests/regression/run_regression_tests.sh
```

## Test Categories

### Core Language Tests
- **Basic Syntax**: Pattern declarations, variable assignments, expressions
- **Control Flow**: if/else statements, while/for loops
- **Functions**: User-defined function declarations and calls
- **Pattern Inheritance**: Pattern composition and inheritance
- **Exception Handling**: try/catch/finally/throw constructs
- **Async/Await**: Asynchronous pattern execution

### Built-in Function Tests
- **Math Functions**: Trigonometric, exponential, logarithmic, power functions
- **Statistical Functions**: mean, median, stddev, correlation, percentile
- **Array Support**: Array literals, access, mixed types
- **Time Series**: Moving averages, trend detection, rate of change
- **Data Transformation**: Normalization, filtering, scaling, clamping
- **Pattern Matching**: Regex and fuzzy string matching
- **Aggregation**: Grouping, pivoting, rollup operations

### Performance Tests
- **Computation Speed**: Large mathematical calculations
- **Memory Usage**: Array handling and large datasets
- **Function Call Overhead**: Repeated function invocations

### Edge Case Tests
- **Boundary Values**: Zero, negative numbers, empty inputs
- **Error Conditions**: Invalid arguments, domain errors
- **Type Consistency**: Mixed number/string/boolean operations
- **Resource Management**: Memory allocation/deallocation

## Test Files

### Main Test Suite
- `regression_test_suite.sep` - Comprehensive test covering all language features
- `run_regression_tests.sh` - Automated test runner with reporting

### Individual Tests
Generated dynamically by the test runner:
- `basic_syntax_test.sep` - Core syntax verification
- `math_test.sep` - Mathematical function testing
- `stats_test.sep` - Statistical function testing
- `array_test.sep` - Array support verification
- `control_flow_test.sep` - Control structure testing
- `function_test.sep` - User function testing
- `exception_test.sep` - Exception handling testing
- `performance_test.sep` - Performance benchmarking

## Test Output

All test results are saved to `tests/regression/output/`:
- `*_output.txt` - Standard output from each test
- `*_error.txt` - Error messages and warnings
- `*_perf.txt` - Performance timing results

## Expected Results

### Passing Tests
✅ All core language features should execute without errors
✅ Mathematical calculations should produce correct results
✅ Statistical functions should handle edge cases gracefully
✅ Performance tests should complete within reasonable time limits

### Acceptable Warnings
⚠️ Engine integration warnings when quantum functions unavailable
⚠️ Precision warnings for floating-point calculations
⚠️ Memory usage warnings for large datasets

### Failure Conditions
❌ Syntax errors in previously valid code
❌ Incorrect mathematical results
❌ Function signature changes breaking existing code
❌ Performance regressions > 50% slowdown
❌ Memory leaks or excessive resource usage

## Adding New Regression Tests

### For New Language Features
1. Add test cases to `regression_test_suite.sep`
2. Verify tests pass with current version
3. Document expected behavior in comments

### For Bug Fixes
1. Create specific test reproducing the original bug
2. Verify test fails with buggy version
3. Verify test passes with fix applied
4. Add to permanent regression suite

### For Performance Improvements
1. Add performance test with measurable benchmark
2. Document acceptable performance ranges
3. Set up automated performance monitoring

## Continuous Integration

The regression test suite integrates with GitHub Actions:

```yaml
- name: Run Regression Tests
  run: |
    ./build.sh
    ./tests/regression/run_regression_tests.sh
```

Tests run automatically on:
- Pull requests to main branch
- Release tag creation
- Nightly builds for performance monitoring

## Debugging Failed Tests

### 1. Check Error Logs
```bash
# View error details
cat tests/regression/output/*_error.txt

# Check specific test output
cat tests/regression/output/math_test_output.txt
```

### 2. Run Individual Tests
```bash
# Run single test manually
./build/src/dsl/sep_dsl_interpreter tests/regression/output/math_test.sep

# Run with verbose output
./build/src/dsl/sep_dsl_interpreter --verbose tests/regression/output/math_test.sep
```

### 3. Compare with Previous Versions
```bash
# Test with different DSL versions
git checkout v1.1.0
./build.sh
./build/src/dsl/sep_dsl_interpreter tests/regression/output/math_test.sep

git checkout main
./build.sh
./build/src/dsl/sep_dsl_interpreter tests/regression/output/math_test.sep
```

### 4. Performance Analysis
```bash
# Profile performance test
valgrind --tool=callgrind ./build/src/dsl/sep_dsl_interpreter tests/regression/output/performance_test.sep

# Check memory usage
valgrind --tool=memcheck ./build/src/dsl/sep_dsl_interpreter tests/regression/output/performance_test.sep
```

## Version Compatibility Matrix

| DSL Version | Test Suite Version | Compatibility | Notes |
|-------------|-------------------|---------------|-------|
| 1.2.0       | 1.0               | ✅ Full       | Current version |
| 1.1.0       | 1.0               | ✅ Full       | All tests pass |
| 1.0.0       | 1.0               | ⚠️ Partial    | Some async tests fail |

## Maintenance

### Regular Tasks
- Run regression tests before each release
- Update performance benchmarks quarterly
- Review and clean up test output files monthly
- Validate test coverage against new features

### Annual Tasks
- Performance baseline review and updates
- Test suite architecture review
- Platform compatibility verification
- Historical version compatibility audit

---

*This regression test suite is critical for maintaining SEP DSL quality and backward compatibility. All developers should run these tests before submitting changes that could affect language behavior.*
