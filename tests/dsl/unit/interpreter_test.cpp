#include <gtest/gtest.h>
#include "dsl/runtime/interpreter.h"
#include "dsl/parser/parser.h"
#include <any>

using namespace dsl::runtime;
using namespace dsl::parser;

class InterpreterTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper to run simple DSL code and check it doesn't crash
    void runCode(const std::string& code) {
        Parser parser(code);
        auto program = parser.parse();
        
        Interpreter interpreter;
        // This should not throw any exceptions
        ASSERT_NO_THROW(interpreter.interpret(*program));
    }
};

TEST_F(InterpreterTest, ExecutesSimpleAssignment) {
    runCode("pattern test { x = 42; }");
}

TEST_F(InterpreterTest, ExecutesBooleanAssignment) {
    runCode("pattern test { flag = true; other = false; }");
}

TEST_F(InterpreterTest, ExecutesArithmeticOperations) {
    runCode("pattern test { result = 5 + 3; }");
    runCode("pattern test { result = 10 - 4; }");
    runCode("pattern test { result = 6 * 7; }");
    runCode("pattern test { result = 15 / 3; }");
}

TEST_F(InterpreterTest, ExecutesComparisonOperations) {
    runCode("pattern test { result = 5 > 3; }");
    runCode("pattern test { result = 5 < 10; }");
    runCode("pattern test { result = 5 == 5; }");
    runCode("pattern test { result = 5 != 3; }");
    runCode("pattern test { result = 5 >= 3; }");
    runCode("pattern test { result = 3 <= 5; }");
}

TEST_F(InterpreterTest, ExecutesLogicalOperations) {
    runCode("pattern test { result = true && false; }");
    runCode("pattern test { result = true || false; }");
}

TEST_F(InterpreterTest, ExecutesIfStatement) {
    runCode("pattern test { if (true) { x = 1; } else { x = 0; } }");
    runCode("pattern test { if (5 > 3) { x = 10; } }");
}

TEST_F(InterpreterTest, ExecutesWhileLoop) {
    runCode("pattern test { x = 0; while (x < 3) { x = x + 1; } }");
}

TEST_F(InterpreterTest, ExecutesFunctionDeclaration) {
    runCode("pattern test { function add(a, b) { return a + b; } }");
}

TEST_F(InterpreterTest, ExecutesFunctionCall) {
    runCode("pattern test { function double_it(n) { return n * 2; } result = double_it(5); }");
}

TEST_F(InterpreterTest, ExecutesBuiltinFunctionCall) {
    runCode("pattern test { result = abs(-5); }");
}

TEST_F(InterpreterTest, ExecutesComplexExpression) {
    runCode("pattern test { result = 2 + 3 * 4; }");
    runCode("pattern test { result = (2 + 3) * 4; }");
}

TEST_F(InterpreterTest, ExecutesNestedBlocks) {
    runCode(R"(
        pattern test { 
            x = 0;
            while (x < 2) {
                if (x == 0) {
                    y = 10;
                } else {
                    y = 20;
                }
                x = x + 1;
            }
        }
    )");
}

TEST_F(InterpreterTest, ExecutesFunctionWithLocalVariables) {
    runCode(R"(
        pattern test {
            function calculate(n) {
                temp = n * 2;
                result = temp + 1;
                return result;
            }
            final = calculate(5);
        }
    )");
}

TEST_F(InterpreterTest, ExecutesRecursiveFunction) {
    runCode(R"(
        pattern test {
            function factorial(n) {
                if (n <= 1) {
                    return 1;
                } else {
                    return n * factorial(n - 1);
                }
            }
            result = factorial(5);
        }
    )");
}

TEST_F(InterpreterTest, HandlesVariableScoping) {
    runCode(R"(
        pattern test {
            x = 10;
            function test_scope() {
                x = 20;
                return x;
            }
            result = test_scope();
        }
    )");
}

TEST_F(InterpreterTest, ExecutesMultiplePatterns) {
    runCode(R"(
        pattern first {
            x = 5;
        }
        pattern second {
            y = 10;
        }
    )");
}

TEST_F(InterpreterTest, HandlesEmptyPattern) {
    runCode("pattern empty { }");
}

TEST_F(InterpreterTest, HandlesComplexArithmetic) {
    runCode("pattern test { result = ((2 + 3) * 4 - 1) / 2; }");
}

TEST_F(InterpreterTest, HandlesStringLiterals) {
    runCode("pattern test { message = \"hello world\"; }");
}

TEST_F(InterpreterTest, ExecutesBuiltinMathFunctions) {
    runCode("pattern test { result1 = abs(-10); result2 = sqrt(16); result3 = sin(0); }");
}

// Comprehensive error handling tests
TEST_F(InterpreterTest, HandlesUndefinedVariables) {
    EXPECT_THROW({
        runCode("pattern test { x = undefined_variable; }");
    }, std::runtime_error);
}

TEST_F(InterpreterTest, HandlesUndefinedFunctions) {
    EXPECT_THROW({
        runCode("pattern test { x = unknown_function(5); }");
    }, std::runtime_error);
}

TEST_F(InterpreterTest, HandlesWrongArgumentCounts) {
    // abs() expects 1 argument
    EXPECT_THROW({
        runCode("pattern test { x = abs(1, 2); }");
    }, std::runtime_error);
    
    // min() expects 2 arguments
    EXPECT_THROW({
        runCode("pattern test { x = min(5); }");
    }, std::runtime_error);
}

TEST_F(InterpreterTest, HandlesNegativeSqrt) {
    EXPECT_THROW({
        runCode("pattern test { x = sqrt(-1); }");
    }, std::runtime_error);
}

TEST_F(InterpreterTest, ExecutesPrintFunction) {
    // Test that print function executes without throwing
    EXPECT_NO_THROW({
        runCode("pattern test { print(\"Hello\", 42, true); }");
    });
}
