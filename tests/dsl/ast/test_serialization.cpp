#include <gtest/gtest.h>
#include "dsl/ast/serializer.h"
#include "dsl/parser/parser.h"

using namespace dsl::ast;
using namespace dsl::parser;

class ASTSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple test DSL program - only patterns
        test_dsl_source = R"(
pattern test {
    result = 3 * 4 + 2
    print("Test:", result)
    
    if (result > 10) {
        status = "high"
    } else {
        status = "low"
    }
}
)";
    }
    
    std::string test_dsl_source;
};

TEST_F(ASTSerializationTest, BasicSerialization) {
    // Parse the DSL source
    Parser parser(test_dsl_source);
    auto original_program = parser.parse();
    
    // Serialize to JSON
    ASTSerializer serializer;
    auto json = serializer.serialize(*original_program);
    
    // Check that JSON contains expected structure
    EXPECT_EQ(json["type"].get<std::string>(), "Program");
    EXPECT_TRUE(json.contains("patterns"));
    EXPECT_TRUE(json.contains("streams"));
    EXPECT_TRUE(json.contains("signals"));
    
    // Check that we have the expected number of each element
    EXPECT_EQ(json["patterns"].size(), 1);
    EXPECT_EQ(json["streams"].size(), 0);  // No streams in this test
    EXPECT_EQ(json["signals"].size(), 0);  // No signals in this test
}

TEST_F(ASTSerializationTest, RoundtripSerialization) {
    // Parse the DSL source
    Parser parser(test_dsl_source);
    auto original_program = parser.parse();
    
    // Serialize to JSON
    ASTSerializer serializer;
    auto json = serializer.serialize(*original_program);
    
    // Deserialize back to AST
    auto restored_program = serializer.deserialize(json);
    
    // Verify the restored program has the same structure
    EXPECT_EQ(restored_program->patterns.size(), original_program->patterns.size());
    EXPECT_EQ(restored_program->streams.size(), original_program->streams.size());
    EXPECT_EQ(restored_program->signals.size(), original_program->signals.size());
    
    // Check pattern details
    EXPECT_EQ(restored_program->patterns[0]->name, original_program->patterns[0]->name);
    EXPECT_EQ(restored_program->patterns[0]->body.size(), original_program->patterns[0]->body.size());
}

TEST_F(ASTSerializationTest, FileOperations) {
    // Parse the DSL source
    Parser parser(test_dsl_source);
    auto original_program = parser.parse();
    
    // Save to file
    ASTSerializer serializer;
    std::string test_file = "/tmp/test_ast.json";
    EXPECT_TRUE(serializer.saveToFile(*original_program, test_file));
    
    // Load from file
    auto loaded_program = serializer.loadFromFile(test_file);
    
    // Verify loaded program matches original
    EXPECT_EQ(loaded_program->patterns.size(), original_program->patterns.size());
    EXPECT_EQ(loaded_program->streams.size(), original_program->streams.size());
    EXPECT_EQ(loaded_program->signals.size(), original_program->signals.size());
    
    // Clean up
    std::remove(test_file.c_str());
}

TEST_F(ASTSerializationTest, ComplexExpressionSerialization) {
    std::string complex_source = R"(
pattern complex {
    // Binary operations
    math = (3 + 4) * (5 - 2)
    
    // Function calls
    result = measure_entropy("test")
    
    // Array operations
    arr = [1, 2, 3, 4]
    first = arr[0]
    
    // Unary operations
    neg = -5
    not_val = !true
}
)";
    
    Parser parser(complex_source);
    auto original_program = parser.parse();
    
    ASTSerializer serializer;
    auto json = serializer.serialize(*original_program);
    auto restored_program = serializer.deserialize(json);
    
    // Verify the complex expressions are preserved
    EXPECT_EQ(restored_program->patterns.size(), 1);
    EXPECT_GT(restored_program->patterns[0]->body.size(), 4); // Should have multiple statements
}

TEST_F(ASTSerializationTest, SimplePatternSerialization) {
    std::string simple_source = R"(
pattern simple {
    value = 3 + 4
    result = measure_entropy("test")
}
)";
    
    Parser parser(simple_source);
    auto original_program = parser.parse();
    
    ASTSerializer serializer;
    auto json = serializer.serialize(*original_program);
    auto restored_program = serializer.deserialize(json);
    
    // Check that pattern is preserved
    EXPECT_EQ(restored_program->patterns.size(), 1);
    EXPECT_GT(restored_program->patterns[0]->body.size(), 0);
}

TEST_F(ASTSerializationTest, ErrorHandling) {
    ASTSerializer serializer;
    
    // Test invalid JSON
    nlohmann::json invalid_json;
    invalid_json["type"] = "InvalidType";
    
    EXPECT_THROW(serializer.deserialize(invalid_json), std::runtime_error);
    
    // Test nonexistent file
    EXPECT_THROW(serializer.loadFromFile("/nonexistent/path/file.json"), std::runtime_error);
}
