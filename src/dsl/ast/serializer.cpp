#include "serializer.h"
#include <fstream>
#include <iostream>
#include <typeinfo>

namespace dsl::ast {

json ASTSerializer::serialize_source_location(const SourceLocation& loc) {
    return json{
        {"line", loc.line},
        {"column", loc.column}
    };
}

SourceLocation ASTSerializer::deserialize_source_location(const json& j) {
    SourceLocation loc;
    if (j.contains("line")) {
        loc.line = j["line"].get<size_t>();
    }
    if (j.contains("column")) {
        loc.column = j["column"].get<size_t>();
    }
    return loc;
}

json ASTSerializer::serialize_type_annotation(TypeAnnotation type) {
    switch (type) {
        case TypeAnnotation::NUMBER: return "number";
        case TypeAnnotation::STRING: return "string";
        case TypeAnnotation::BOOL: return "bool";
        case TypeAnnotation::PATTERN: return "pattern";
        case TypeAnnotation::VOID: return "void";
        case TypeAnnotation::ARRAY: return "array";
        case TypeAnnotation::INFERRED: return "inferred";
        default: return "unknown";
    }
}

TypeAnnotation ASTSerializer::deserialize_type_annotation(const json& j) {
    std::string type_str = j.get<std::string>();
    if (type_str == "number") return TypeAnnotation::NUMBER;
    if (type_str == "string") return TypeAnnotation::STRING;
    if (type_str == "bool") return TypeAnnotation::BOOL;
    if (type_str == "pattern") return TypeAnnotation::PATTERN;
    if (type_str == "void") return TypeAnnotation::VOID;
    if (type_str == "array") return TypeAnnotation::ARRAY;
    return TypeAnnotation::INFERRED;
}

json ASTSerializer::serialize(const Expression& expr) {
    json j;
    j["location"] = serialize_source_location(expr.location);
    
    // Use dynamic_cast to determine the actual type
    if (const auto* num = dynamic_cast<const NumberLiteral*>(&expr)) {
        j["type"] = "NumberLiteral";
        j["value"] = num->value;
    }
    else if (const auto* str = dynamic_cast<const StringLiteral*>(&expr)) {
        j["type"] = "StringLiteral";
        j["value"] = str->value;
    }
    else if (const auto* boolean = dynamic_cast<const BooleanLiteral*>(&expr)) {
        j["type"] = "BooleanLiteral";
        j["value"] = boolean->value;
    }
    else if (const auto* id = dynamic_cast<const Identifier*>(&expr)) {
        j["type"] = "Identifier";
        j["name"] = id->name;
    }
    else if (const auto* binop = dynamic_cast<const BinaryOp*>(&expr)) {
        j["type"] = "BinaryOp";
        j["op"] = binop->op;
        j["left"] = serialize(*binop->left);
        j["right"] = serialize(*binop->right);
    }
    else if (const auto* unop = dynamic_cast<const UnaryOp*>(&expr)) {
        j["type"] = "UnaryOp";
        j["op"] = unop->op;
        j["right"] = serialize(*unop->right);
    }
    else if (const auto* call = dynamic_cast<const Call*>(&expr)) {
        j["type"] = "Call";
        j["callee"] = call->callee;
        j["args"] = json::array();
        for (const auto& arg : call->args) {
            j["args"].push_back(serialize(*arg));
        }
    }
    else if (const auto* member = dynamic_cast<const MemberAccess*>(&expr)) {
        j["type"] = "MemberAccess";
        j["object"] = serialize(*member->object);
        j["member"] = member->member;
    }
    else if (const auto* array = dynamic_cast<const ArrayLiteral*>(&expr)) {
        j["type"] = "ArrayLiteral";
        j["elements"] = json::array();
        for (const auto& elem : array->elements) {
            j["elements"].push_back(serialize(*elem));
        }
    }
    else if (const auto* access = dynamic_cast<const ArrayAccess*>(&expr)) {
        j["type"] = "ArrayAccess";
        j["array"] = serialize(*access->array);
        j["index"] = serialize(*access->index);
    }
    else if (const auto* await_expr = dynamic_cast<const AwaitExpression*>(&expr)) {
        j["type"] = "AwaitExpression";
        j["expression"] = serialize(*await_expr->expression);
    }
    else if (const auto* weighted = dynamic_cast<const WeightedSum*>(&expr)) {
        j["type"] = "WeightedSum";
        j["pairs"] = json::array();
        for (const auto& pair : weighted->pairs) {
            json pair_json;
            pair_json["weight"] = serialize(*pair.first);
            pair_json["value"] = serialize(*pair.second);
            j["pairs"].push_back(pair_json);
        }
    }
    else {
        j["type"] = "Unknown";
    }
    
    return j;
}

json ASTSerializer::serialize(const Statement& stmt) {
    json j;
    j["location"] = serialize_source_location(stmt.location);
    
    if (const auto* assign = dynamic_cast<const Assignment*>(&stmt)) {
        j["type"] = "Assignment";
        j["name"] = assign->name;
        j["type_annotation"] = serialize_type_annotation(assign->type);
        j["value"] = serialize(*assign->value);
    }
    else if (const auto* expr_stmt = dynamic_cast<const ExpressionStatement*>(&stmt)) {
        j["type"] = "ExpressionStatement";
        j["expression"] = serialize(*expr_stmt->expression);
    }
    else if (const auto* if_stmt = dynamic_cast<const IfStatement*>(&stmt)) {
        j["type"] = "IfStatement";
        j["condition"] = serialize(*if_stmt->condition);
        
        j["then_branch"] = json::array();
        for (const auto& stmt : if_stmt->then_branch) {
            j["then_branch"].push_back(serialize(*stmt));
        }
        
        j["else_branch"] = json::array();
        for (const auto& stmt : if_stmt->else_branch) {
            j["else_branch"].push_back(serialize(*stmt));
        }
    }
    else if (const auto* while_stmt = dynamic_cast<const WhileStatement*>(&stmt)) {
        j["type"] = "WhileStatement";
        j["condition"] = serialize(*while_stmt->condition);
        
        j["body"] = json::array();
        for (const auto& stmt : while_stmt->body) {
            j["body"].push_back(serialize(*stmt));
        }
    }
    else if (const auto* for_stmt = dynamic_cast<const ForStatement*>(&stmt)) {
        j["type"] = "ForStatement";
        j["variable"] = for_stmt->variable;
        j["iterable"] = serialize(*for_stmt->iterable);
        
        j["body"] = json::array();
        for (const auto& stmt : for_stmt->body) {
            j["body"].push_back(serialize(*stmt));
        }
    }
    else if (const auto* ret_stmt = dynamic_cast<const ReturnStatement*>(&stmt)) {
        j["type"] = "ReturnStatement";
        if (ret_stmt->value) {
            j["value"] = serialize(*ret_stmt->value);
        }
    }
    else if (const auto* func_decl = dynamic_cast<const FunctionDeclaration*>(&stmt)) {
        j["type"] = "FunctionDeclaration";
        j["name"] = func_decl->name;
        j["return_type"] = serialize_type_annotation(func_decl->return_type);
        
        j["parameters"] = json::array();
        for (const auto& param : func_decl->parameters) {
            json param_json;
            param_json["name"] = param.first;
            param_json["type"] = serialize_type_annotation(param.second);
            j["parameters"].push_back(param_json);
        }
        
        j["body"] = json::array();
        for (const auto& stmt : func_decl->body) {
            j["body"].push_back(serialize(*stmt));
        }
    }
    else if (const auto* async_func = dynamic_cast<const AsyncFunctionDeclaration*>(&stmt)) {
        j["type"] = "AsyncFunctionDeclaration";
        j["name"] = async_func->name;
        j["return_type"] = serialize_type_annotation(async_func->return_type);
        
        j["parameters"] = json::array();
        for (const auto& param : async_func->parameters) {
            json param_json;
            param_json["name"] = param.first;
            param_json["type"] = serialize_type_annotation(param.second);
            j["parameters"].push_back(param_json);
        }
        
        j["body"] = json::array();
        for (const auto& stmt : async_func->body) {
            j["body"].push_back(serialize(*stmt));
        }
    }
    else if (const auto* try_stmt = dynamic_cast<const TryStatement*>(&stmt)) {
        j["type"] = "TryStatement";
        j["catch_variable"] = try_stmt->catch_variable;
        
        j["try_body"] = json::array();
        for (const auto& stmt : try_stmt->try_body) {
            j["try_body"].push_back(serialize(*stmt));
        }
        
        j["catch_body"] = json::array();
        for (const auto& stmt : try_stmt->catch_body) {
            j["catch_body"].push_back(serialize(*stmt));
        }
        
        j["finally_body"] = json::array();
        for (const auto& stmt : try_stmt->finally_body) {
            j["finally_body"].push_back(serialize(*stmt));
        }
    }
    else if (const auto* throw_stmt = dynamic_cast<const ThrowStatement*>(&stmt)) {
        j["type"] = "ThrowStatement";
        j["expression"] = serialize(*throw_stmt->expression);
    }
    else {
        j["type"] = "Unknown";
    }
    
    return j;
}

json ASTSerializer::serialize(const Program& program) {
    json j;
    j["type"] = "Program";
    j["location"] = serialize_source_location(program.location);
    
    j["streams"] = json::array();
    for (const auto& stream : program.streams) {
        json stream_json;
        stream_json["type"] = "StreamDecl";
        stream_json["location"] = serialize_source_location(stream->location);
        stream_json["name"] = stream->name;
        stream_json["source"] = stream->source;
        
        stream_json["params"] = json::object();
        for (const auto& param : stream->params) {
            stream_json["params"][param.first] = param.second;
        }
        j["streams"].push_back(stream_json);
    }
    
    j["patterns"] = json::array();
    for (const auto& pattern : program.patterns) {
        json pattern_json;
        pattern_json["type"] = "PatternDecl";
        pattern_json["location"] = serialize_source_location(pattern->location);
        pattern_json["name"] = pattern->name;
        pattern_json["parent_pattern"] = pattern->parent_pattern;
        
        pattern_json["inputs"] = json::array();
        for (const auto& input : pattern->inputs) {
            pattern_json["inputs"].push_back(input);
        }
        
        pattern_json["body"] = json::array();
        for (const auto& stmt : pattern->body) {
            pattern_json["body"].push_back(serialize(*stmt));
        }
        j["patterns"].push_back(pattern_json);
    }
    
    j["signals"] = json::array();
    for (const auto& signal : program.signals) {
        json signal_json;
        signal_json["type"] = "SignalDecl";
        signal_json["location"] = serialize_source_location(signal->location);
        signal_json["name"] = signal->name;
        signal_json["action"] = signal->action;
        
        if (signal->trigger) {
            signal_json["trigger"] = serialize(*signal->trigger);
        }
        if (signal->confidence) {
            signal_json["confidence"] = serialize(*signal->confidence);
        }
        j["signals"].push_back(signal_json);
    }
    
    return j;
}

bool ASTSerializer::save_to_file(const Program& program, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }
    
    json j = serialize(program);
    file << j.dump(2);  // Pretty print with 2-space indentation
    
    return true;
}

std::unique_ptr<Program> ASTSerializer::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading." << std::endl;
        return nullptr;
    }
    
    json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return nullptr;
    }
    
    return deserialize_program(j);
}

// Basic deserialization implementation (simplified for now)
std::unique_ptr<Program> ASTSerializer::deserialize_program(const json& j) {
    auto program = std::make_unique<Program>();
    
    if (j.contains("location")) {
        program->location = deserialize_source_location(j["location"]);
    }
    
    // For now, we'll implement basic deserialization
    // Full implementation would deserialize all streams, patterns, and signals
    // This is a foundation that can be extended
    
    return program;
}

std::unique_ptr<Expression> ASTSerializer::deserialize_expression(const json& j) {
    // Basic implementation - can be extended for full deserialization
    if (!j.contains("type")) {
        return nullptr;
    }
    
    std::string type = j["type"].get<std::string>();
    
    if (type == "NumberLiteral") {
        auto expr = std::make_unique<NumberLiteral>();
        expr->value = j["value"].get<double>();
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    // Add more expression types as needed
    
    return nullptr;
}

std::unique_ptr<Statement> ASTSerializer::deserialize_statement(const json& j) {
    // Basic implementation - can be extended for full deserialization
    if (!j.contains("type")) {
        return nullptr;
    }
    
    std::string type = j["type"].get<std::string>();
    
    if (type == "Assignment") {
        auto stmt = std::make_unique<Assignment>();
        stmt->name = j["name"].get<std::string>();
        if (j.contains("type_annotation")) {
            stmt->type = deserialize_type_annotation(j["type_annotation"]);
        }
        if (j.contains("location")) {
            stmt->location = deserialize_source_location(j["location"]);
        }
        // Note: value deserialization would need to be implemented
        return stmt;
    }
    // Add more statement types as needed
    
    return nullptr;
}

} // namespace dsl::ast
