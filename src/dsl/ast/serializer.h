#pragma once
#include "nodes.h"
#include <json/json.h>
#include <memory>

namespace dsl::ast {

class ASTSerializer {
public:
    // Serialize AST to JSON
    static Json::Value serialize(const Node& node);
    static Json::Value serialize(const Program& program);
    static Json::Value serialize(const Expression& expr);
    static Json::Value serialize(const Statement& stmt);
    
    // Deserialize JSON to AST
    static std::unique_ptr<Program> deserialize_program(const Json::Value& json);
    static std::unique_ptr<Expression> deserialize_expression(const Json::Value& json);
    static std::unique_ptr<Statement> deserialize_statement(const Json::Value& json);
    
    // File I/O helpers
    static bool save_to_file(const Program& program, const std::string& filename);
    static std::unique_ptr<Program> load_from_file(const std::string& filename);

private:
    // Helper methods for specific node types
    static Json::Value serialize_source_location(const SourceLocation& loc);
    static SourceLocation deserialize_source_location(const Json::Value& json);
    
    static Json::Value serialize_type_annotation(TypeAnnotation type);
    static TypeAnnotation deserialize_type_annotation(const Json::Value& json);
};

} // namespace dsl::ast
