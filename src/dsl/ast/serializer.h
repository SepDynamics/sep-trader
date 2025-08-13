#include "nlohmann_json_safe.h"
#pragma once
#include <array>
#include <memory>

#include "nodes.h"

namespace dsl::ast {

using json = nlohmann::json;

class ASTSerializer {
public:
    // Serialize AST to JSON
    static json serialize(const Node& node);
    static json serialize(const Program& program);
    static json serialize(const Expression& expr);
    static json serialize(const Statement& stmt);
    
    // Deserialize JSON to AST
    static std::unique_ptr<Program> deserialize_program(const json& j);
    static std::unique_ptr<Expression> deserialize_expression(const json& j);
    static std::unique_ptr<Statement> deserialize_statement(const json& j);
    
    // File I/O helpers
    static bool save_to_file(const Program& program, const std::string& filename);
    static std::unique_ptr<Program> load_from_file(const std::string& filename);

private:
    // Helper methods for specific node types
    static json serialize_source_location(const SourceLocation& loc);
    static SourceLocation deserialize_source_location(const json& j);
    
    static json serialize_type_annotation(TypeAnnotation type);
    static TypeAnnotation deserialize_type_annotation(const json& j);
};

} // namespace dsl::ast
