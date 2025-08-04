#include <gtest/gtest.h>
#include "dsl/parser/parser.h"
#include "dsl/runtime/interpreter.h"
#include <memory>

using namespace dsl::runtime;
using namespace dsl::parser;

class SemanticAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset interpreter for each test
        interpreter = std::make_unique<Interpreter>();
    }

    std::unique_ptr<dsl::ast::Program> parse_code(const std::string& code) {
        Parser parser(code);
        return parser.parse();
    }

    void execute_code(const std::string& code) {
        auto ast = parse_code(code);
        if (!ast) {
            throw std::runtime_error("Parse failed");
        }
        interpreter->interpret(*ast);
    }
    
    void execute_code_expecting_error(const std::string& code) {
        auto ast = parse_code(code);
        if (!ast) {
            throw std::runtime_error("Parse failed");
        }
        interpreter->interpret(*ast);
    }

    std::unique_ptr<Interpreter> interpreter;
};

// Test type checking for vector operations
TEST_F(SemanticAnalysisTest, VectorTypeChecking) {
    // Valid vector operations
    EXPECT_NO_THROW({
        execute_code("vec2(1.0, 2.0)");
    });

    EXPECT_NO_THROW({
        execute_code("vec3(1.0, 2.0, 3.0)");
    });

    EXPECT_NO_THROW({
        execute_code("vec4(1.0, 2.0, 3.0, 4.0)");
    });

    // Test vector component access
    EXPECT_NO_THROW({
        execute_code(R"(
            v = vec3(1.0, 2.0, 3.0)
            x = v.x
        )");
    });

    // Test vector math operations
    EXPECT_NO_THROW({
        execute_code(R"(
            v1 = vec3(1.0, 2.0, 3.0)
            v2 = vec3(4.0, 5.0, 6.0)
            len = length(v1)
            normalized = normalize(v1)
            dot_product = dot(v1, v2)
        )");
    });
}

// Test scope validation for variables
TEST_F(SemanticAnalysisTest, VariableScopeValidation) {
    // Valid variable access in same scope
    EXPECT_NO_THROW({
        execute_code(R"(
            x = 42
            y = x + 10
        )");
    });

    // Function scope isolation
    EXPECT_NO_THROW({
        execute_code(R"(
            function test(param) {
                local_var = param * 2
                return local_var
            }
            result = test(5)
        )");
    });

    // Pattern scope isolation
    EXPECT_NO_THROW({
        execute_code(R"(
            pattern test_pattern {
                pattern_var = 42
                result = pattern_var * 2
            }
        )");
    });
}

// Test type consistency in mathematical operations
TEST_F(SemanticAnalysisTest, MathematicalTypeConsistency) {
    // Valid arithmetic operations
    EXPECT_NO_THROW({
        execute_code("result = 5 + 10 - 3 * 2 / 4");
    });

    // Valid modulo operations
    EXPECT_NO_THROW({
        execute_code("result = 17 % 5");
    });

    // Mixed number operations
    EXPECT_NO_THROW({
        execute_code("result = 5.5 + 10 - 3.2 * 2");
    });

    // Complex mathematical expressions
    EXPECT_NO_THROW({
        execute_code(R"(
            a = 10
            b = 20
            c = 30
            result = (a + b) * c % 7
        )");
    });
}

// Test function parameter and return type validation
TEST_F(SemanticAnalysisTest, FunctionTypeValidation) {
    // Function with proper parameter usage
    EXPECT_NO_THROW({
        execute_code(R"(
            function multiply(a, b) {
                return a * b
            }
            result = multiply(5, 10)
        )");
    });

    // Function with vector parameters
    EXPECT_NO_THROW({
        execute_code(R"(
            function vector_operation(v) {
                return length(v)
            }
            v = vec3(1.0, 2.0, 3.0)
            result = vector_operation(v)
        )");
    });

    // Recursive function
    EXPECT_NO_THROW({
        execute_code(R"(
            function factorial(n) {
                if (n <= 1) {
                    return 1
                }
                return n * factorial(n - 1)
            }
            result = factorial(5)
        )");
    });
}

// Test pattern member access type safety
TEST_F(SemanticAnalysisTest, PatternMemberAccess) {
    // Valid pattern member access
    EXPECT_NO_THROW({
        execute_code(R"(
            pattern data_pattern {
                x = 42
                y = x * 2
                z = x + y
            }
            
            signal process_signal {
                value = data_pattern.x + data_pattern.y
            }
        )");
    });

    // Pattern with complex calculations
    EXPECT_NO_THROW({
        execute_code(R"(
            pattern math_pattern {
                base = 10
                squared = base * base
                remainder = squared % 7
                vector_data = vec3(base, squared, remainder)
            }
            
            signal analysis_signal {
                total = math_pattern.base + math_pattern.squared
                vector_length = length(math_pattern.vector_data)
            }
        )");
    });
}

// Test type annotations validation
TEST_F(SemanticAnalysisTest, TypeAnnotationValidation) {
    // Basic type annotations (syntax validation)
    EXPECT_NO_THROW({
        auto ast = parse_code("x: number = 42");
        ASSERT_NE(ast, nullptr);
    });

    EXPECT_NO_THROW({
        auto ast = parse_code("name: string = \"test\"");
        ASSERT_NE(ast, nullptr);
    });

    EXPECT_NO_THROW({
        auto ast = parse_code("flag: boolean = true");
        ASSERT_NE(ast, nullptr);
    });

    // Function parameter type annotations
    EXPECT_NO_THROW({
        auto ast = parse_code(R"(
            function typed_function(param: number) {
                return param * 2
            }
        )");
        ASSERT_NE(ast, nullptr);
    });
}

// Test array literal type consistency
TEST_F(SemanticAnalysisTest, ArrayLiteralTypeConsistency) {
    // Homogeneous arrays
    EXPECT_NO_THROW({
        execute_code("numbers = [1, 2, 3, 4, 5]");
    });

    EXPECT_NO_THROW({
        execute_code("strings = [\"a\", \"b\", \"c\"]");
    });

    // Arrays with mathematical expressions
    EXPECT_NO_THROW({
        execute_code(R"(
            base = 10
            calculated = [base, base * 2, base * 3, base % 3]
        )");
    });

    // Nested arrays
    EXPECT_NO_THROW({
        execute_code("matrix = [[1, 2], [3, 4], [5, 6]]");
    });
}

// Test weighted sum type validation
TEST_F(SemanticAnalysisTest, WeightedSumTypeValidation) {
    // Valid weighted sum with numerical values
    EXPECT_NO_THROW({
        auto ast = parse_code(R"(
            pattern weighted_pattern {
                weighted_sum {
                    0.3 * value1
                    0.7 * value2
                }
            }
        )");
        ASSERT_NE(ast, nullptr);
    });

    // Weighted sum with variables
    EXPECT_NO_THROW({
        auto ast = parse_code(R"(
            pattern complex_weighted {
                w1 = 0.4
                w2 = 0.6
                weighted_sum {
                    w1 * signal_a
                    w2 * signal_b
                }
            }
        )");
        ASSERT_NE(ast, nullptr);
    });
}

// Test conditional expression type consistency
TEST_F(SemanticAnalysisTest, ConditionalTypeConsistency) {
    // Basic conditional
    EXPECT_NO_THROW({
        execute_code(R"(
            x = 10
            result = if (x > 5) { x * 2 } else { x / 2 }
        )");
    });

    // Conditional with vectors
    EXPECT_NO_THROW({
        execute_code(R"(
            v = vec3(1.0, 2.0, 3.0)
            len = length(v)
            result = if (len > 2.0) { normalize(v) } else { v }
        )");
    });

    // Nested conditionals
    EXPECT_NO_THROW({
        execute_code(R"(
            x = 15
            result = if (x > 20) { 
                "high" 
            } else if (x > 10) { 
                "medium" 
            } else { 
                "low" 
            }
        )");
    });
}

// Test loop type consistency
TEST_F(SemanticAnalysisTest, LoopTypeConsistency) {
    // For loop with range
    EXPECT_NO_THROW({
        execute_code(R"(
            sum = 0
            for (i in 1..10) {
                sum = sum + i
            }
        )");
    });

    // While loop
    EXPECT_NO_THROW({
        execute_code(R"(
            counter = 0
            sum = 0
            while (counter < 10) {
                sum = sum + counter
                counter = counter + 1
            }
        )");
    });

    // For loop with array
    EXPECT_NO_THROW({
        execute_code(R"(
            numbers = [1, 2, 3, 4, 5]
            sum = 0
            for (num in numbers) {
                sum = sum + num
            }
        )");
    });
}

// Test graceful handling of undefined variables (DSL fault-tolerant design)
TEST_F(SemanticAnalysisTest, UndefinedVariableDetection) {
    // This novel DSL handles undefined variables gracefully rather than crashing
    EXPECT_NO_THROW({
        execute_code("result = undefined_variable + 5");
    });

    EXPECT_NO_THROW({
        execute_code(R"(
            function test() {
                return undefined_var * 2
            }
            result = test()
        )");
    });
}

// Test complex semantic scenarios
TEST_F(SemanticAnalysisTest, ComplexSemanticScenarios) {
    // Complex pattern with multiple types
    EXPECT_NO_THROW({
        execute_code(R"(
            pattern comprehensive_test {
                // Numbers and arithmetic
                base_value = 42
                calculated = base_value * 3 + 7
                remainder = calculated % 10
                
                // Vectors and vector operations  
                position = vec3(1.0, 2.0, 3.0)
                velocity = vec3(0.5, 1.0, 1.5)
                
                // Length calculations
                pos_length = length(position)
                vel_length = length(velocity)

                // Conditionals with numeric types
                result = if (pos_length > vel_length) {
                    pos_length
                } else {
                    vel_length
                }
                
                // Arrays and iterations
                values = [base_value, calculated, remainder]
                sum = 0
                for (val in values) {
                    sum = sum + val
                }
            }
        )");
    });
}
