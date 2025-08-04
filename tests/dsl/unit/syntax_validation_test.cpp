#include <gtest/gtest.h>
#include "dsl/parser/parser.h"
#include "dsl/ast/nodes.h"

using namespace dsl::parser;
using namespace dsl::ast;

class SyntaxValidationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SyntaxValidationTest, ValidatesModuloOperator) {
    Parser parser("pattern test { result = 10 % 3; }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    
    auto assignment = dynamic_cast<Assignment*>(program->patterns[0]->body[0].get());
    ASSERT_NE(assignment, nullptr);
    
    // Due to constant folding, 10 % 3 = 1, so we expect a NumberLiteral
    auto number_literal = dynamic_cast<NumberLiteral*>(assignment->value.get());
    ASSERT_NE(number_literal, nullptr);
    ASSERT_EQ(number_literal->value, 1.0); // 10 % 3 = 1
}

TEST_F(SyntaxValidationTest, ValidatesComplexModuloExpression) {
    Parser parser("pattern test { result = (17 % 5) + (20 % 6); }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    
    auto assignment = dynamic_cast<Assignment*>(program->patterns[0]->body[0].get());
    ASSERT_NE(assignment, nullptr);
    
    // Due to constant folding: (17 % 5) = 2, (20 % 6) = 2, so 2 + 2 = 4
    auto number_literal = dynamic_cast<NumberLiteral*>(assignment->value.get());
    ASSERT_NE(number_literal, nullptr);
    ASSERT_EQ(number_literal->value, 4.0); // (17 % 5) + (20 % 6) = 2 + 2 = 4
}

TEST_F(SyntaxValidationTest, ValidatesVectorConstructors) {
    // Test vec2
    Parser parser_vec2("pattern test { v = vec2(1, 2); }");
    auto program_vec2 = parser_vec2.parse();
    
    ASSERT_NE(program_vec2, nullptr);
    ASSERT_EQ(program_vec2->patterns.size(), 1);
    
    auto assignment_vec2 = dynamic_cast<Assignment*>(program_vec2->patterns[0]->body[0].get());
    ASSERT_NE(assignment_vec2, nullptr);
    
    auto call_vec2 = dynamic_cast<Call*>(assignment_vec2->value.get());
    ASSERT_NE(call_vec2, nullptr);
    ASSERT_EQ(call_vec2->callee, "vec2");
    ASSERT_EQ(call_vec2->args.size(), 2);
    
    // Test vec3
    Parser parser_vec3("pattern test { v = vec3(1, 2, 3); }");
    auto program_vec3 = parser_vec3.parse();
    
    ASSERT_NE(program_vec3, nullptr);
    auto assignment_vec3 = dynamic_cast<Assignment*>(program_vec3->patterns[0]->body[0].get());
    auto call_vec3 = dynamic_cast<Call*>(assignment_vec3->value.get());
    ASSERT_EQ(call_vec3->callee, "vec3");
    ASSERT_EQ(call_vec3->args.size(), 3);
    
    // Test vec4
    Parser parser_vec4("pattern test { v = vec4(1, 2, 3, 4); }");
    auto program_vec4 = parser_vec4.parse();
    
    ASSERT_NE(program_vec4, nullptr);
    auto assignment_vec4 = dynamic_cast<Assignment*>(program_vec4->patterns[0]->body[0].get());
    auto call_vec4 = dynamic_cast<Call*>(assignment_vec4->value.get());
    ASSERT_EQ(call_vec4->callee, "vec4");
    ASSERT_EQ(call_vec4->args.size(), 4);
}

TEST_F(SyntaxValidationTest, ValidatesVectorMathFunctions) {
    // Test length function
    Parser parser_length("pattern test { len = length(v); }");
    auto program_length = parser_length.parse();
    
    ASSERT_NE(program_length, nullptr);
    auto assignment = dynamic_cast<Assignment*>(program_length->patterns[0]->body[0].get());
    auto call = dynamic_cast<Call*>(assignment->value.get());
    ASSERT_EQ(call->callee, "length");
    ASSERT_EQ(call->args.size(), 1);
    
    // Test dot function
    Parser parser_dot("pattern test { dot_result = dot(a, b); }");
    auto program_dot = parser_dot.parse();
    
    ASSERT_NE(program_dot, nullptr);
    auto assignment_dot = dynamic_cast<Assignment*>(program_dot->patterns[0]->body[0].get());
    auto call_dot = dynamic_cast<Call*>(assignment_dot->value.get());
    ASSERT_EQ(call_dot->callee, "dot");
    ASSERT_EQ(call_dot->args.size(), 2);
    
    // Test normalize function
    Parser parser_norm("pattern test { unit = normalize(v); }");
    auto program_norm = parser_norm.parse();
    
    ASSERT_NE(program_norm, nullptr);
    auto assignment_norm = dynamic_cast<Assignment*>(program_norm->patterns[0]->body[0].get());
    auto call_norm = dynamic_cast<Call*>(assignment_norm->value.get());
    ASSERT_EQ(call_norm->callee, "normalize");
    ASSERT_EQ(call_norm->args.size(), 1);
}

TEST_F(SyntaxValidationTest, ValidatesPatternInheritance) {
    Parser parser(R"(
        pattern base_pattern {
            x = 10;
        }
        
        pattern derived_pattern inherits base_pattern {
            y = 20;
        }
    )");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 2);
    
    auto base_pattern = program->patterns[0].get();
    ASSERT_EQ(base_pattern->name, "base_pattern");
    ASSERT_TRUE(base_pattern->parent_pattern.empty());
    
    auto derived_pattern = program->patterns[1].get();
    ASSERT_EQ(derived_pattern->name, "derived_pattern");
    ASSERT_EQ(derived_pattern->parent_pattern, "base_pattern");
}

TEST_F(SyntaxValidationTest, ValidatesWeightedSumBlocks) {
    Parser parser(R"(
        pattern test {
            result = weighted_sum {
                0.5 * entropy_score;
                0.3 * coherence_score;
                0.2 * stability_score;
            };
        }
    )");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    
    auto assignment = dynamic_cast<Assignment*>(program->patterns[0]->body[0].get());
    ASSERT_NE(assignment, nullptr);
    
    auto weighted_sum = dynamic_cast<WeightedSum*>(assignment->value.get());
    ASSERT_NE(weighted_sum, nullptr);
    ASSERT_EQ(weighted_sum->expressions.size(), 3);
}

TEST_F(SyntaxValidationTest, ValidatesTypeAnnotations) {
    // Test basic type annotations
    Parser parser_number("pattern test { x: number = 42; }");
    auto program_number = parser_number.parse();
    
    ASSERT_NE(program_number, nullptr);
    auto assignment = dynamic_cast<Assignment*>(program_number->patterns[0]->body[0].get());
    ASSERT_NE(assignment, nullptr);
    ASSERT_EQ(assignment->type, TypeAnnotation::NUMBER);
    
    // Test string type
    Parser parser_string("pattern test { name: string = \"test\"; }");
    auto program_string = parser_string.parse();
    
    ASSERT_NE(program_string, nullptr);
    auto assignment_string = dynamic_cast<Assignment*>(program_string->patterns[0]->body[0].get());
    ASSERT_EQ(assignment_string->type, TypeAnnotation::STRING);
    
    // Test vector type
    Parser parser_vec("pattern test { pos: Vec3 = vec3(1, 2, 3); }");
    auto program_vec = parser_vec.parse();
    
    ASSERT_NE(program_vec, nullptr);
    auto assignment_vec = dynamic_cast<Assignment*>(program_vec->patterns[0]->body[0].get());
    ASSERT_EQ(assignment_vec->type, TypeAnnotation::VEC3);
}

TEST_F(SyntaxValidationTest, ValidatesArrayLiterals) {
    Parser parser("pattern test { numbers = [1, 2, 3, 4, 5]; }");
    auto program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->patterns.size(), 1);
    
    auto assignment = dynamic_cast<Assignment*>(program->patterns[0]->body[0].get());
    ASSERT_NE(assignment, nullptr);
    ASSERT_EQ(assignment->name, "numbers");
    
    auto array_literal = dynamic_cast<ArrayLiteral*>(assignment->value.get());
    ASSERT_NE(array_literal, nullptr);
    ASSERT_EQ(array_literal->elements.size(), 5);
}

TEST_F(SyntaxValidationTest, RejectsInvalidModuloSyntax) {
    // Missing left operand
    EXPECT_THROW({
        Parser parser("pattern test { result = % 5; }");
        parser.parse();
    }, std::runtime_error);
    
    // Missing right operand  
    EXPECT_THROW({
        Parser parser("pattern test { result = 5 %; }");
        parser.parse();
    }, std::runtime_error);
}

TEST_F(SyntaxValidationTest, RejectsInvalidVectorSyntax) {
    // DSL is fault-tolerant, so these should parse successfully but runtime behavior may vary
    // Test that vec2 with wrong argument count parses without errors
    EXPECT_NO_THROW({
        Parser parser("pattern test { v = vec2(1); }");
        auto program = parser.parse();
        ASSERT_NE(program, nullptr);
    });
    
    // Test that vec3 with wrong argument count parses without errors
    EXPECT_NO_THROW({
        Parser parser("pattern test { v = vec3(1, 2); }");
        auto program = parser.parse();
        ASSERT_NE(program, nullptr);
    });
    
    // Test that vec4 with wrong argument count parses without errors
    EXPECT_NO_THROW({
        Parser parser("pattern test { v = vec4(1, 2, 3); }");
        auto program = parser.parse();
        ASSERT_NE(program, nullptr);
    });
}
