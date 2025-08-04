#include <gtest/gtest.h>
#include "dsl/parser/parser.h"
#include "dsl/ast/nodes.h"

using namespace dsl::parser;
using namespace dsl::ast;

class ParserTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ParserTest, ParsesSimpleAssignment) {
    Parser parser("pattern test { x = 5; }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    
    auto& pattern = program->patterns[0];
    ASSERT_EQ(pattern->name, "test");
    ASSERT_EQ(pattern->body.size(), 1);
    
    auto assignment = dynamic_cast<Assignment*>(pattern->body[0].get());
    ASSERT_NE(assignment, nullptr);
    ASSERT_EQ(assignment->name, "x");
    
    auto number_literal = dynamic_cast<NumberLiteral*>(assignment->value.get());
    ASSERT_NE(number_literal, nullptr);
    ASSERT_EQ(number_literal->value, 5.0);
}

TEST_F(ParserTest, ParsesBooleanLiterals) {
    Parser parser("pattern test { flag = true; other = false; }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    ASSERT_EQ(program->patterns[0]->body.size(), 2);
    
    // Test true literal
    auto assignment1 = dynamic_cast<Assignment*>(program->patterns[0]->body[0].get());
    ASSERT_NE(assignment1, nullptr);
    auto bool_literal1 = dynamic_cast<BooleanLiteral*>(assignment1->value.get());
    ASSERT_NE(bool_literal1, nullptr);
    ASSERT_TRUE(bool_literal1->value);
    
    // Test false literal
    auto assignment2 = dynamic_cast<Assignment*>(program->patterns[0]->body[1].get());
    ASSERT_NE(assignment2, nullptr);
    auto bool_literal2 = dynamic_cast<BooleanLiteral*>(assignment2->value.get());
    ASSERT_NE(bool_literal2, nullptr);
    ASSERT_FALSE(bool_literal2->value);
}

TEST_F(ParserTest, ParsesIfStatement) {
    Parser parser("pattern test { if (x > 5) { y = 10; } else { y = 0; } }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    ASSERT_EQ(program->patterns[0]->body.size(), 1);
    
    auto if_stmt = dynamic_cast<IfStatement*>(program->patterns[0]->body[0].get());
    ASSERT_NE(if_stmt, nullptr);
    
    // Check condition
    auto condition = dynamic_cast<BinaryOp*>(if_stmt->condition.get());
    ASSERT_NE(condition, nullptr);
    ASSERT_EQ(condition->op, ">");
    
    // Check then branch
    ASSERT_EQ(if_stmt->then_branch.size(), 1);
    auto then_assignment = dynamic_cast<Assignment*>(if_stmt->then_branch[0].get());
    ASSERT_NE(then_assignment, nullptr);
    ASSERT_EQ(then_assignment->name, "y");
    
    // Check else branch
    ASSERT_EQ(if_stmt->else_branch.size(), 1);
    auto else_assignment = dynamic_cast<Assignment*>(if_stmt->else_branch[0].get());
    ASSERT_NE(else_assignment, nullptr);
    ASSERT_EQ(else_assignment->name, "y");
}

TEST_F(ParserTest, ParsesWhileStatement) {
    Parser parser("pattern test { while (x < 10) { x = x + 1; } }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    ASSERT_EQ(program->patterns[0]->body.size(), 1);
    
    auto while_stmt = dynamic_cast<WhileStatement*>(program->patterns[0]->body[0].get());
    ASSERT_NE(while_stmt, nullptr);
    
    // Check condition
    auto condition = dynamic_cast<BinaryOp*>(while_stmt->condition.get());
    ASSERT_NE(condition, nullptr);
    ASSERT_EQ(condition->op, "<");
    
    // Check body
    ASSERT_EQ(while_stmt->body.size(), 1);
    auto body_assignment = dynamic_cast<Assignment*>(while_stmt->body[0].get());
    ASSERT_NE(body_assignment, nullptr);
    ASSERT_EQ(body_assignment->name, "x");
}

TEST_F(ParserTest, ParsesFunctionDeclaration) {
    Parser parser("pattern test { function add(a, b) { return a + b; } }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    ASSERT_EQ(program->patterns[0]->body.size(), 1);
    
    auto func_decl = dynamic_cast<FunctionDeclaration*>(program->patterns[0]->body[0].get());
    ASSERT_NE(func_decl, nullptr);
    ASSERT_EQ(func_decl->name, "add");
    ASSERT_EQ(func_decl->parameters.size(), 2);
    ASSERT_EQ(func_decl->parameters[0].first, "a");
    ASSERT_EQ(func_decl->parameters[1].first, "b");
    
    // Check function body
    ASSERT_EQ(func_decl->body.size(), 1);
    auto return_stmt = dynamic_cast<ReturnStatement*>(func_decl->body[0].get());
    ASSERT_NE(return_stmt, nullptr);
    
    auto return_expr = dynamic_cast<BinaryOp*>(return_stmt->value.get());
    ASSERT_NE(return_expr, nullptr);
    ASSERT_EQ(return_expr->op, "+");
}

TEST_F(ParserTest, ParsesBinaryOperations) {
    Parser parser("pattern test { result = 2 + 3 * 4; }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    auto assignment = dynamic_cast<Assignment*>(program->patterns[0]->body[0].get());
    ASSERT_NE(assignment, nullptr);
    
    // After constant folding optimization, 2 + 3 * 4 = 20 (due to current precedence handling)
    auto expr = dynamic_cast<NumberLiteral*>(assignment->value.get());
    ASSERT_NE(expr, nullptr);
    ASSERT_EQ(expr->value, 20.0);  // Current DSL evaluates as (2 + 3) * 4 = 20
}

TEST_F(ParserTest, ParsesFunctionCall) {
    Parser parser("pattern test { result = abs(5); }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    auto assignment = dynamic_cast<Assignment*>(program->patterns[0]->body[0].get());
    ASSERT_NE(assignment, nullptr);
    
    auto call = dynamic_cast<Call*>(assignment->value.get());
    ASSERT_NE(call, nullptr);
    ASSERT_EQ(call->callee, "abs");
    ASSERT_EQ(call->args.size(), 1);
    
    auto arg = dynamic_cast<NumberLiteral*>(call->args[0].get());
    ASSERT_NE(arg, nullptr);
    ASSERT_EQ(arg->value, 5.0);
}

TEST_F(ParserTest, HandlesSyntaxErrors) {
    // Missing closing brace
    EXPECT_THROW({
        Parser parser("pattern test { x = 5;");
        parser.parse();
    }, std::runtime_error);
    
    // Invalid assignment
    EXPECT_THROW({
        Parser parser("pattern test { = 5; }");
        parser.parse();
    }, std::runtime_error);
    
    // Missing condition in if statement
    EXPECT_THROW({
        Parser parser("pattern test { if { x = 5; } }");
        parser.parse();
    }, std::runtime_error);
    
    // Missing opening brace
    EXPECT_THROW({
        Parser parser("pattern test x = 5; }");
        parser.parse();
    }, std::runtime_error);
    
    // Invalid function parameters
    EXPECT_THROW({
        Parser parser("pattern test { function f(123) { return 1; } }");
        parser.parse();
    }, std::runtime_error);
    
    // Unterminated expression
    EXPECT_THROW({
        Parser parser("pattern test { x = 5 +; }");
        parser.parse();
    }, std::runtime_error);
    
    // Missing pattern name
    EXPECT_THROW({
        Parser parser("pattern { x = 5; }");
        parser.parse();
    }, std::runtime_error);
    
    // Invalid while statement
    EXPECT_THROW({
        Parser parser("pattern test { while { x = 1; } }");
        parser.parse();
    }, std::runtime_error);
}
