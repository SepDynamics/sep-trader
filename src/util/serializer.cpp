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
        j["expressions"] = json::array();
        for (const auto& expr : weighted->expressions) {
            j["expressions"].push_back(serialize(*expr));
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

// Full deserialization implementation
std::unique_ptr<Program> ASTSerializer::deserialize_program(const json& j) {
    auto program = std::make_unique<Program>();
    
    if (j.contains("location")) {
        program->location = deserialize_source_location(j["location"]);
    }
    
    // Deserialize patterns
    if (j.contains("patterns") && j["patterns"].is_array()) {
        for (const auto& pattern_json : j["patterns"]) {
            auto pattern = std::make_unique<PatternDecl>();
            
            if (pattern_json.contains("name")) {
                pattern->name = pattern_json["name"].get<std::string>();
            }
            if (pattern_json.contains("parent_pattern")) {
                pattern->parent_pattern = pattern_json["parent_pattern"].get<std::string>();
            }
            if (pattern_json.contains("location")) {
                pattern->location = deserialize_source_location(pattern_json["location"]);
            }
            
            // Deserialize pattern inputs
            if (pattern_json.contains("inputs") && pattern_json["inputs"].is_array()) {
                for (const auto& input : pattern_json["inputs"]) {
                    pattern->inputs.push_back(input.get<std::string>());
                }
            }
            
            // Deserialize pattern body
            if (pattern_json.contains("body") && pattern_json["body"].is_array()) {
                for (const auto& stmt_json : pattern_json["body"]) {
                    auto stmt = deserialize_statement(stmt_json);
                    if (stmt) {
                        pattern->body.push_back(std::move(stmt));
                    }
                }
            }
            
            program->patterns.push_back(std::move(pattern));
        }
    }
    
    // Deserialize streams
    if (j.contains("streams") && j["streams"].is_array()) {
        for (const auto& stream_json : j["streams"]) {
            auto stream = std::make_unique<StreamDecl>();
            
            if (stream_json.contains("name")) {
                stream->name = stream_json["name"].get<std::string>();
            }
            if (stream_json.contains("source")) {
                stream->source = stream_json["source"].get<std::string>();
            }
            if (stream_json.contains("location")) {
                stream->location = deserialize_source_location(stream_json["location"]);
            }
            
            // Deserialize parameters
            if (stream_json.contains("params") && stream_json["params"].is_object()) {
                for (auto& [key, value] : stream_json["params"].items()) {
                    stream->params[key] = value.get<std::string>();
                }
            }
            
            program->streams.push_back(std::move(stream));
        }
    }
    
    // Deserialize signals
    if (j.contains("signals") && j["signals"].is_array()) {
        for (const auto& signal_json : j["signals"]) {
            auto signal = std::make_unique<SignalDecl>();
            
            if (signal_json.contains("name")) {
                signal->name = signal_json["name"].get<std::string>();
            }
            if (signal_json.contains("action")) {
                signal->action = signal_json["action"].get<std::string>();
            }
            if (signal_json.contains("location")) {
                signal->location = deserialize_source_location(signal_json["location"]);
            }
            
            // Deserialize trigger and confidence expressions
            if (signal_json.contains("trigger")) {
                signal->trigger = deserialize_expression(signal_json["trigger"]);
            }
            if (signal_json.contains("confidence")) {
                signal->confidence = deserialize_expression(signal_json["confidence"]);
            }
            
            program->signals.push_back(std::move(signal));
        }
    }
    
    return program;
}

std::unique_ptr<Expression> ASTSerializer::deserialize_expression(const json& j) {
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
    else if (type == "StringLiteral") {
        auto expr = std::make_unique<StringLiteral>();
        expr->value = j["value"].get<std::string>();
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    else if (type == "BooleanLiteral") {
        auto expr = std::make_unique<BooleanLiteral>();
        expr->value = j["value"].get<bool>();
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    else if (type == "Identifier") {
        auto expr = std::make_unique<Identifier>();
        expr->name = j["name"].get<std::string>();
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    else if (type == "BinaryOp") {
        auto expr = std::make_unique<BinaryOp>();
        expr->op = j["op"].get<std::string>();
        if (j.contains("left")) {
            expr->left = deserialize_expression(j["left"]);
        }
        if (j.contains("right")) {
            expr->right = deserialize_expression(j["right"]);
        }
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    else if (type == "UnaryOp") {
        auto expr = std::make_unique<UnaryOp>();
        expr->op = j["op"].get<std::string>();
        if (j.contains("right")) {
            expr->right = deserialize_expression(j["right"]);
        }
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    else if (type == "Call") {
        auto expr = std::make_unique<Call>();
        expr->callee = j["callee"].get<std::string>();
        if (j.contains("args") && j["args"].is_array()) {
            for (const auto& arg_json : j["args"]) {
                auto arg = deserialize_expression(arg_json);
                if (arg) {
                    expr->args.push_back(std::move(arg));
                }
            }
        }
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    else if (type == "ArrayLiteral") {
        auto expr = std::make_unique<ArrayLiteral>();
        if (j.contains("elements") && j["elements"].is_array()) {
            for (const auto& elem_json : j["elements"]) {
                auto elem = deserialize_expression(elem_json);
                if (elem) {
                    expr->elements.push_back(std::move(elem));
                }
            }
        }
        if (j.contains("location")) {
            expr->location = deserialize_source_location(j["location"]);
        }
        return expr;
    }
    
    return nullptr;
}

std::unique_ptr<Statement> ASTSerializer::deserialize_statement(const json& j) {
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
        if (j.contains("value")) {
            stmt->value = deserialize_expression(j["value"]);
        }
        if (j.contains("location")) {
            stmt->location = deserialize_source_location(j["location"]);
        }
        return stmt;
    }
    else if (type == "ExpressionStatement") {
        auto stmt = std::make_unique<ExpressionStatement>();
        if (j.contains("expression")) {
            stmt->expression = deserialize_expression(j["expression"]);
        }
        if (j.contains("location")) {
            stmt->location = deserialize_source_location(j["location"]);
        }
        return stmt;
    }
    else if (type == "IfStatement") {
        auto stmt = std::make_unique<IfStatement>();
        if (j.contains("condition")) {
            stmt->condition = deserialize_expression(j["condition"]);
        }
        if (j.contains("then_branch") && j["then_branch"].is_array()) {
            for (const auto& then_stmt : j["then_branch"]) {
                auto deserialized_stmt = deserialize_statement(then_stmt);
                if (deserialized_stmt) {
                    stmt->then_branch.push_back(std::move(deserialized_stmt));
                }
            }
        }
        if (j.contains("else_branch") && j["else_branch"].is_array()) {
            for (const auto& else_stmt : j["else_branch"]) {
                auto deserialized_stmt = deserialize_statement(else_stmt);
                if (deserialized_stmt) {
                    stmt->else_branch.push_back(std::move(deserialized_stmt));
                }
            }
        }
        if (j.contains("location")) {
            stmt->location = deserialize_source_location(j["location"]);
        }
        return stmt;
    }
    else if (type == "WhileStatement") {
        auto stmt = std::make_unique<WhileStatement>();
        if (j.contains("condition")) {
            stmt->condition = deserialize_expression(j["condition"]);
        }
        if (j.contains("body") && j["body"].is_array()) {
            for (const auto& body_stmt : j["body"]) {
                auto deserialized_stmt = deserialize_statement(body_stmt);
                if (deserialized_stmt) {
                    stmt->body.push_back(std::move(deserialized_stmt));
                }
            }
        }
        if (j.contains("location")) {
            stmt->location = deserialize_source_location(j["location"]);
        }
        return stmt;
    }
    else if (type == "ReturnStatement") {
        auto stmt = std::make_unique<ReturnStatement>();
        if (j.contains("value")) {
            stmt->value = deserialize_expression(j["value"]);
        }
        if (j.contains("location")) {
            stmt->location = deserialize_source_location(j["location"]);
        }
        return stmt;
    }
    
    return nullptr;
}

} // namespace dsl::ast
