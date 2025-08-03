# TASK.md - SEP DSL Next Phase Development

## 🎯 Current Sprint: Core Functionality Implementation

### Phase 2A: Implement Critical Built-in Functions (Priority 1)
*Goal: Make the DSL actually useful by implementing the core built-ins*

#### 1. Core Primitives Module (`src/dsl/stdlib/core_primitives.cpp`)
- [ ] Implement type checking functions
  - [ ] `is_number(value)` - Check if value is numeric
  - [ ] `is_string(value)` - Check if value is string
  - [ ] `is_bool(value)` - Check if value is boolean
  - [ ] `to_string(value)` - Convert any value to string
  - [ ] `to_number(value)` - Convert to number with error handling

#### 2. Math Module (`src/dsl/stdlib/math/math.cpp`)
- [ ] Basic arithmetic functions
  - [ ] `abs(x)` - Absolute value
  - [ ] `min(a, b)` - Minimum of two values
  - [ ] `max(a, b)` - Maximum of two values
  - [ ] `round(x)` - Round to nearest integer
  - [ ] `floor(x)` - Floor function
  - [ ] `ceil(x)` - Ceiling function
- [ ] Trigonometric functions
  - [ ] `sin(x)`, `cos(x)`, `tan(x)`
  - [ ] `asin(x)`, `acos(x)`, `atan(x)`
- [ ] Exponential/logarithmic
  - [ ] `exp(x)` - e^x
  - [ ] `log(x)` - Natural logarithm
  - [ ] `log10(x)` - Base-10 logarithm
  - [ ] `pow(x, y)` - x raised to power y
  - [ ] `sqrt(x)` - Square root

#### 3. Statistical Module (`src/dsl/stdlib/statistical/statistical.cpp`)
- [ ] Basic statistics (operating on arrays/windows)
  - [ ] `mean(data)` - Average value
  - [ ] `median(data)` - Middle value
  - [ ] `std_dev(data)` - Standard deviation
  - [ ] `variance(data)` - Variance
  - [ ] `min_value(data)` - Minimum in dataset
  - [ ] `max_value(data)` - Maximum in dataset
- [ ] Advanced statistics
  - [ ] `correlation(data1, data2)` - Pearson correlation
  - [ ] `covariance(data1, data2)` - Covariance
  - [ ] `percentile(data, p)` - Nth percentile

### Phase 2B: Complete Control Flow Implementation (Priority 2)
*Goal: Enable real programming logic in patterns*

#### 1. Parser Enhancements (`src/dsl/parser/parser.cpp`)
- [ ] Complete `parse_if_statement()` implementation
  - [ ] Handle `else if` chains
  - [ ] Ensure proper block parsing
- [ ] Implement `parse_while_statement()`
  - [ ] Basic while loop structure
  - [ ] Break/continue support
- [ ] Implement `parse_for_statement()`
  - [ ] Traditional for loop: `for (i = 0; i < 10; i++)`
  - [ ] For-each style: `for (value in array)`

#### 2. Interpreter Support (`src/dsl/runtime/interpreter.cpp`)
- [ ] Add visitor methods for control flow
  - [ ] `visit_if_statement()`
  - [ ] `visit_while_statement()`
  - [ ] `visit_for_statement()`
- [ ] Implement break/continue exceptions
- [ ] Add proper scope management for loops

### Phase 2C: Engine Integration Enhancement (Priority 3)
*Goal: Connect DSL built-ins to real quantum analysis*

#### 1. Wire Up Quantum Functions (`src/dsl/runtime/engine_bridge.cpp`)
- [ ] Connect `qfh()` to real Quantum Fourier Heuristic
  - [ ] Pass window data from patterns
  - [ ] Return computed QFH values
- [ ] Connect `qbsa()` to Quantum Behavioral State Analysis
  - [ ] Handle probe/expectation parameters
  - [ ] Return validation scores
- [ ] Implement pattern result caching
  - [ ] Cache computed patterns for performance
  - [ ] Invalidate on data updates

#### 2. Data Window Management
- [ ] Implement window slicing for patterns
  - [ ] `window.slice(start, end)`
  - [ ] `window.last(n)` - Last n values
  - [ ] `window.first(n)` - First n values
- [ ] Add real-time data updates
  - [ ] Stream new data into windows
  - [ ] Trigger pattern re-evaluation

### Phase 2D: Testing Infrastructure (Priority 4)
*Goal: Ensure reliability as we add features*

#### 1. Unit Test Framework Setup
- [ ] Set up Google Test integration
- [ ] Create test structure matching src/
  ```
  tests/
  ├── dsl/
  │   ├── lexer/
  │   ├── parser/
  │   ├── runtime/
  │   └── stdlib/
  ```

#### 2. Core Test Suites
- [ ] Lexer tests
  - [ ] Token recognition for all types
  - [ ] Error cases and edge conditions
- [ ] Parser tests
  - [ ] Expression parsing correctness
  - [ ] Statement parsing validation
  - [ ] Error recovery scenarios
- [ ] Interpreter tests
  - [ ] Expression evaluation
  - [ ] Control flow execution
  - [ ] Built-in function calls

#### 3. Integration Tests
- [ ] End-to-end DSL script execution
- [ ] Pattern → Signal flow validation
- [ ] Engine bridge functionality

## 📋 Implementation Order & Time Estimates

### Week 1: Core Built-ins
1. **Day 1-2**: Implement core_primitives and basic math functions
2. **Day 3-4**: Implement statistical functions
3. **Day 5**: Test and debug built-in functions

### Week 2: Control Flow & Testing
1. **Day 1-2**: Complete parser control flow
2. **Day 3-4**: Implement interpreter control flow
3. **Day 5**: Set up test framework and write initial tests

### Week 3: Engine Integration
1. **Day 1-2**: Wire up quantum functions
2. **Day 3-4**: Implement window management
3. **Day 5**: Integration testing and debugging

## 🚀 Quick Start Commands

```bash
# Build with your new stdlib modules
cd /sep
./build.sh

# Test DSL with new features
./build/src/dsl/main examples/test_builtin_functions.sep

# Run unit tests (once implemented)
./build/tests/dsl_tests

# Interactive REPL testing
./build/src/dsl/repl
```

## 📝 Code Templates

### Adding a Built-in Function (Math Example)
```cpp
// In src/dsl/stdlib/math/math.cpp
void register_math(Runtime& runtime) {
    // Basic math function
    runtime.define_builtin("sqrt", [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("sqrt() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        if (x < 0) {
            throw std::runtime_error("sqrt() of negative number");
        }
        return std::sqrt(x);
    });
    
    // Continue adding more functions...
}
```

### Testing Pattern
```cpp
// In tests/dsl/stdlib/math_test.cpp
TEST(MathFunctions, SquareRoot) {
    std::string source = R"(
        x = sqrt(16)
        y = sqrt(2.25)
    )";
    
    Parser parser(source);
    auto program = parser.parse();
    
    Interpreter interpreter;
    interpreter.interpret(*program);
    
    // Verify results
    EXPECT_DOUBLE_EQ(interpreter.get_variable("x"), 4.0);
    EXPECT_DOUBLE_EQ(interpreter.get_variable("y"), 1.5);
}
```

## 🎯 Success Criteria

### Sprint Complete When:
1. ✅ All basic math and statistical functions work in DSL scripts
2. ✅ If/else and while loops execute correctly
3. ✅ At least 3 quantum functions return real computed values
4. ✅ Unit tests pass for all new functionality
5. ✅ Can run a complex pattern analysis script end-to-end

## 💡 Next Sprint Preview
After completing this sprint, the next focus areas will be:
- Time series specific functions (moving averages, trend detection)
- Pattern composition and inheritance
- Performance optimization (bytecode compilation)
- Developer tools (LSP implementation)

---

*Remember: Focus on getting a few things working really well rather than many things working poorly. Each completed function is a step toward the commercial-grade DSL!*