#include "serializer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace dsl::ast {

nlohmann::json ASTSerializer::serialize(const Program& program) {
    nlohmann::json root;
    root["type"] = "Program";
    root["location"] = serializeSourceLocation(program.location);
    
    nlohmann::json streams = nlohmann::json::array();
    for (const auto& stream : program.streams) {
        streams.push_back(serializeStreamDecl(*stream));
    }
    root["streams"] = streams;
    
    nlohmann::json patterns = nlohmann::json::array();
    for (const auto& pattern : program.patterns) {
        patterns.push_back(serializePatternDecl(*pattern));
    }
    root["patterns"] = patterns;
    
    nlohmann::json signals = nlohmann::json::array();
    for (const auto& signal : program.signals) {
        signals.push_back(serializeSignalDecl(*signal));
    }
    root["signals"] = signals;
    
    return root;
}

std::unique_ptr<Program> ASTSerializer::deserialize(const nlohmann::json& json) {
    if (json["type"].get<std::string>() != "Program") {
        throw std::runtime_error("Expected Program node type");
    }
    
    auto program = std::make_unique<Program>();
    program->location = deserializeSourceLocation(json["location"]);
    
    for (const auto& stream_json : json["streams"]) {
        program->streams.push_back(deserializeStreamDecl(stream_json));
    }
    
    for (const auto& pattern_json : json["patterns"]) {
        program->patterns.push_back(deserializePatternDecl(pattern_json));
    }
    
    for (const auto& signal_json : json["signals"]) {
        program->signals.push_back(deserializeSignalDecl(signal_json));
    }
    
    return program;
}

bool ASTSerializer::saveToFile(const Program& program, const std::string& filename) {
    try {
        nlohmann::json json = serialize(program);
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << json.dump(2); // Pretty print with 2 spaces
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving AST to file: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<Program> ASTSerializer::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    nlohmann::json json;
    file >> json;
    
    return deserialize(json);
}

// Expression serialization
nlohmann::json ASTSerializer::serializeExpression(const Expression& expr) {
    // Use dynamic_cast to determine the actual type
    if (auto* numberLit = dynamic_cast<const NumberLiteral*>(&expr)) {
        return serializeNumberLiteral(*numberLit);
    } else if (auto* stringLit = dynamic_cast<const StringLiteral*>(&expr)) {
        return serializeStringLiteral(*stringLit);
    } else if (auto* boolLit = dynamic_cast<const BooleanLiteral*>(&expr)) {
        return serializeBooleanLiteral(*boolLit);
    } else if (auto* identifier = dynamic_cast<const Identifier*>(&expr)) {
        return serializeIdentifier(*identifier);
    } else if (auto* binaryOp = dynamic_cast<const BinaryOp*>(&expr)) {
        return serializeBinaryOp(*binaryOp);
    } else if (auto* unaryOp = dynamic_cast<const UnaryOp*>(&expr)) {
        return serializeUnaryOp(*unaryOp);
    } else if (auto* call = dynamic_cast<const Call*>(&expr)) {
        return serializeCall(*call);
    } else if (auto* memberAccess = dynamic_cast<const MemberAccess*>(&expr)) {
        return serializeMemberAccess(*memberAccess);
    } else if (auto* arrayLit = dynamic_cast<const ArrayLiteral*>(&expr)) {
        return serializeArrayLiteral(*arrayLit);
    } else if (auto* arrayAccess = dynamic_cast<const ArrayAccess*>(&expr)) {
        return serializeArrayAccess(*arrayAccess);
    } else if (auto* weightedSum = dynamic_cast<const WeightedSum*>(&expr)) {
        return serializeWeightedSum(*weightedSum);
    } else if (auto* awaitExpr = dynamic_cast<const AwaitExpression*>(&expr)) {
        return serializeAwaitExpression(*awaitExpr);
    } else {
        throw std::runtime_error("Unknown expression type in serialization");
    }
}

std::unique_ptr<Expression> ASTSerializer::deserializeExpression(const nlohmann::json& json) {
    std::string type = json["type"].get<std::string>();
    
    if (type == "NumberLiteral") {
        return deserializeNumberLiteral(json);
    } else if (type == "StringLiteral") {
        return deserializeStringLiteral(json);
    } else if (type == "BooleanLiteral") {
        return deserializeBooleanLiteral(json);
    } else if (type == "Identifier") {
        return deserializeIdentifier(json);
    } else if (type == "BinaryOp") {
        return deserializeBinaryOp(json);
    } else if (type == "UnaryOp") {
        return deserializeUnaryOp(json);
    } else if (type == "Call") {
        return deserializeCall(json);
    } else if (type == "MemberAccess") {
        return deserializeMemberAccess(json);
    } else if (type == "ArrayLiteral") {
        return deserializeArrayLiteral(json);
    } else if (type == "ArrayAccess") {
        return deserializeArrayAccess(json);
    } else if (type == "WeightedSum") {
        return deserializeWeightedSum(json);
    } else if (type == "AwaitExpression") {
        return deserializeAwaitExpression(json);
    } else {
        throw std::runtime_error("Unknown expression type in deserialization: " + type);
    }
}

// Statement serialization
nlohmann::json ASTSerializer::serializeStatement(const Statement& stmt) {
    if (auto* assignment = dynamic_cast<const Assignment*>(&stmt)) {
        return serializeAssignment(*assignment);
    } else if (auto* exprStmt = dynamic_cast<const ExpressionStatement*>(&stmt)) {
        return serializeExpressionStatement(*exprStmt);
    } else if (auto* evolveStmt = dynamic_cast<const EvolveStatement*>(&stmt)) {
        return serializeEvolveStatement(*evolveStmt);
    } else if (auto* ifStmt = dynamic_cast<const IfStatement*>(&stmt)) {
        return serializeIfStatement(*ifStmt);
    } else if (auto* whileStmt = dynamic_cast<const WhileStatement*>(&stmt)) {
        return serializeWhileStatement(*whileStmt);
    } else if (auto* returnStmt = dynamic_cast<const ReturnStatement*>(&stmt)) {
        return serializeReturnStatement(*returnStmt);
    } else if (auto* funcDecl = dynamic_cast<const FunctionDeclaration*>(&stmt)) {
        return serializeFunctionDeclaration(*funcDecl);
    } else if (auto* asyncFuncDecl = dynamic_cast<const AsyncFunctionDeclaration*>(&stmt)) {
        return serializeAsyncFunctionDeclaration(*asyncFuncDecl);
    } else if (auto* importStmt = dynamic_cast<const ImportStatement*>(&stmt)) {
        return serializeImportStatement(*importStmt);
    } else if (auto* exportStmt = dynamic_cast<const ExportStatement*>(&stmt)) {
        return serializeExportStatement(*exportStmt);
    } else if (auto* tryStmt = dynamic_cast<const TryStatement*>(&stmt)) {
        return serializeTryStatement(*tryStmt);
    } else if (auto* throwStmt = dynamic_cast<const ThrowStatement*>(&stmt)) {
        return serializeThrowStatement(*throwStmt);
    } else {
        throw std::runtime_error("Unknown statement type in serialization");
    }
}

std::unique_ptr<Statement> ASTSerializer::deserializeStatement(const nlohmann::json& json) {
    std::string type = json["type"].get<std::string>();
    
    if (type == "Assignment") {
        return deserializeAssignment(json);
    } else if (type == "ExpressionStatement") {
        return deserializeExpressionStatement(json);
    } else if (type == "EvolveStatement") {
        return deserializeEvolveStatement(json);
    } else if (type == "IfStatement") {
        return deserializeIfStatement(json);
    } else if (type == "WhileStatement") {
        return deserializeWhileStatement(json);
    } else if (type == "ReturnStatement") {
        return deserializeReturnStatement(json);
    } else if (type == "FunctionDeclaration") {
        return deserializeFunctionDeclaration(json);
    } else if (type == "AsyncFunctionDeclaration") {
        return deserializeAsyncFunctionDeclaration(json);
    } else if (type == "ImportStatement") {
        return deserializeImportStatement(json);
    } else if (type == "ExportStatement") {
        return deserializeExportStatement(json);
    } else if (type == "TryStatement") {
        return deserializeTryStatement(json);
    } else if (type == "ThrowStatement") {
        return deserializeThrowStatement(json);
    } else {
        throw std::runtime_error("Unknown statement type in deserialization: " + type);
    }
}

// Expression serialization implementations
nlohmann::json ASTSerializer::serializeNumberLiteral(const NumberLiteral& node) {
    nlohmann::json json;
    json["type"] = "NumberLiteral";
    json["location"] = serializeSourceLocation(node.location);
    json["value"] = node.value;
    return json;
}

nlohmann::json ASTSerializer::serializeStringLiteral(const StringLiteral& node) {
    nlohmann::json json;
    json["type"] = "StringLiteral";
    json["location"] = serializeSourceLocation(node.location);
    json["value"] = node.value;
    return json;
}

nlohmann::json ASTSerializer::serializeBooleanLiteral(const BooleanLiteral& node) {
    nlohmann::json json;
    json["type"] = "BooleanLiteral";
    json["location"] = serializeSourceLocation(node.location);
    json["value"] = node.value;
    return json;
}

nlohmann::json ASTSerializer::serializeIdentifier(const Identifier& node) {
    nlohmann::json json;
    json["type"] = "Identifier";
    json["location"] = serializeSourceLocation(node.location);
    json["name"] = node.name;
    return json;
}

nlohmann::json ASTSerializer::serializeBinaryOp(const BinaryOp& node) {
    nlohmann::json json;
    json["type"] = "BinaryOp";
    json["location"] = serializeSourceLocation(node.location);
    json["left"] = serializeExpression(*node.left);
    json["op"] = node.op;
    json["right"] = serializeExpression(*node.right);
    return json;
}

nlohmann::json ASTSerializer::serializeUnaryOp(const UnaryOp& node) {
    nlohmann::json json;
    json["type"] = "UnaryOp";
    json["location"] = serializeSourceLocation(node.location);
    json["op"] = node.op;
    json["right"] = serializeExpression(*node.right);
    return json;
}

nlohmann::json ASTSerializer::serializeCall(const Call& node) {
    nlohmann::json json;
    json["type"] = "Call";
    json["location"] = serializeSourceLocation(node.location);
    json["callee"] = node.callee;
    json["args"] = serializeExpressions(node.args);
    return json;
}

nlohmann::json ASTSerializer::serializeMemberAccess(const MemberAccess& node) {
    nlohmann::json json;
    json["type"] = "MemberAccess";
    json["location"] = serializeSourceLocation(node.location);
    json["object"] = serializeExpression(*node.object);
    json["member"] = node.member;
    return json;
}

nlohmann::json ASTSerializer::serializeArrayLiteral(const ArrayLiteral& node) {
    nlohmann::json json;
    json["type"] = "ArrayLiteral";
    json["location"] = serializeSourceLocation(node.location);
    json["elements"] = serializeExpressions(node.elements);
    return json;
}

nlohmann::json ASTSerializer::serializeArrayAccess(const ArrayAccess& node) {
    nlohmann::json json;
    json["type"] = "ArrayAccess";
    json["location"] = serializeSourceLocation(node.location);
    json["array"] = serializeExpression(*node.array);
    json["index"] = serializeExpression(*node.index);
    return json;
}

nlohmann::json ASTSerializer::serializeWeightedSum(const WeightedSum& node) {
    nlohmann::json json;
    json["type"] = "WeightedSum";
    json["location"] = serializeSourceLocation(node.location);
    nlohmann::json pairs = nlohmann::json::array();
    for (const auto& pair : node.pairs) {
        nlohmann::json pairJson;
        pairJson["weight"] = serializeExpression(*pair.first);
        pairJson["value"] = serializeExpression(*pair.second);
        pairs.push_back(pairJson);
    }
    json["pairs"] = pairs;
    return json;
}

nlohmann::json ASTSerializer::serializeAwaitExpression(const AwaitExpression& node) {
    nlohmann::json json;
    json["type"] = "AwaitExpression";
    json["location"] = serializeSourceLocation(node.location);
    json["expression"] = serializeExpression(*node.expression);
    return json;
}

// Statement serialization implementations
nlohmann::json ASTSerializer::serializeAssignment(const Assignment& node) {
    nlohmann::json json;
    json["type"] = "Assignment";
    json["location"] = serializeSourceLocation(node.location);
    json["name"] = node.name;
    json["type_annotation"] = serializeTypeAnnotation(node.type);
    json["value"] = serializeExpression(*node.value);
    return json;
}

nlohmann::json ASTSerializer::serializeExpressionStatement(const ExpressionStatement& node) {
    nlohmann::json json;
    json["type"] = "ExpressionStatement";
    json["location"] = serializeSourceLocation(node.location);
    json["expression"] = serializeExpression(*node.expression);
    return json;
}

nlohmann::json ASTSerializer::serializeEvolveStatement(const EvolveStatement& node) {
    nlohmann::json json;
    json["type"] = "EvolveStatement";
    json["location"] = serializeSourceLocation(node.location);
    json["condition"] = serializeExpression(*node.condition);
    json["body"] = serializeStatements(node.body);
    return json;
}

nlohmann::json ASTSerializer::serializeIfStatement(const IfStatement& node) {
    nlohmann::json json;
    json["type"] = "IfStatement";
    json["location"] = serializeSourceLocation(node.location);
    json["condition"] = serializeExpression(*node.condition);
    json["then_branch"] = serializeStatements(node.then_branch);
    json["else_branch"] = serializeStatements(node.else_branch);
    return json;
}

nlohmann::json ASTSerializer::serializeWhileStatement(const WhileStatement& node) {
    nlohmann::json json;
    json["type"] = "WhileStatement";
    json["location"] = serializeSourceLocation(node.location);
    json["condition"] = serializeExpression(*node.condition);
    json["body"] = serializeStatements(node.body);
    return json;
}

nlohmann::json ASTSerializer::serializeReturnStatement(const ReturnStatement& node) {
    nlohmann::json json;
    json["type"] = "ReturnStatement";
    json["location"] = serializeSourceLocation(node.location);
    if (node.value) {
        json["value"] = serializeExpression(*node.value);
    }
    return json;
}

nlohmann::json ASTSerializer::serializeFunctionDeclaration(const FunctionDeclaration& node) {
    nlohmann::json json;
    json["type"] = "FunctionDeclaration";
    json["location"] = serializeSourceLocation(node.location);
    json["name"] = node.name;
    json["return_type"] = serializeTypeAnnotation(node.return_type);
    
    nlohmann::json params = nlohmann::json::array();
    for (const auto& param : node.parameters) {
        nlohmann::json paramJson;
        paramJson["name"] = param.first;
        paramJson["type"] = serializeTypeAnnotation(param.second);
        params.push_back(paramJson);
    }
    json["parameters"] = params;
    json["body"] = serializeStatements(node.body);
    return json;
}

nlohmann::json ASTSerializer::serializeAsyncFunctionDeclaration(const AsyncFunctionDeclaration& node) {
    nlohmann::json json;
    json["type"] = "AsyncFunctionDeclaration";
    json["location"] = serializeSourceLocation(node.location);
    json["name"] = node.name;
    json["return_type"] = serializeTypeAnnotation(node.return_type);
    
    nlohmann::json params = nlohmann::json::array();
    for (const auto& param : node.parameters) {
        nlohmann::json paramJson;
        paramJson["name"] = param.first;
        paramJson["type"] = serializeTypeAnnotation(param.second);
        params.push_back(paramJson);
    }
    json["parameters"] = params;
    json["body"] = serializeStatements(node.body);
    return json;
}

nlohmann::json ASTSerializer::serializeImportStatement(const ImportStatement& node) {
    nlohmann::json json;
    json["type"] = "ImportStatement";
    json["location"] = serializeSourceLocation(node.location);
    json["module_path"] = node.module_path;
    nlohmann::json imports = nlohmann::json::array();
    for (const auto& import : node.imports) {
        imports.push_back(import);
    }
    json["imports"] = imports;
    return json;
}

nlohmann::json ASTSerializer::serializeExportStatement(const ExportStatement& node) {
    nlohmann::json json;
    json["type"] = "ExportStatement";
    json["location"] = serializeSourceLocation(node.location);
    nlohmann::json exports = nlohmann::json::array();
    for (const auto& export_name : node.exports) {
        exports.push_back(export_name);
    }
    json["exports"] = exports;
    return json;
}

nlohmann::json ASTSerializer::serializeTryStatement(const TryStatement& node) {
    nlohmann::json json;
    json["type"] = "TryStatement";
    json["location"] = serializeSourceLocation(node.location);
    json["try_body"] = serializeStatements(node.try_body);
    json["catch_body"] = serializeStatements(node.catch_body);
    json["catch_variable"] = node.catch_variable;
    json["finally_body"] = serializeStatements(node.finally_body);
    return json;
}

nlohmann::json ASTSerializer::serializeThrowStatement(const ThrowStatement& node) {
    nlohmann::json json;
    json["type"] = "ThrowStatement";
    json["location"] = serializeSourceLocation(node.location);
    json["expression"] = serializeExpression(*node.expression);
    return json;
}

nlohmann::json ASTSerializer::serializeStreamDecl(const StreamDecl& node) {
    nlohmann::json json;
    json["type"] = "StreamDecl";
    json["location"] = serializeSourceLocation(node.location);
    json["name"] = node.name;
    json["source"] = node.source;
    
    nlohmann::json params = nlohmann::json::object();
    for (const auto& param : node.params) {
        params[param.first] = param.second;
    }
    json["params"] = params;
    return json;
}

nlohmann::json ASTSerializer::serializePatternDecl(const PatternDecl& node) {
    nlohmann::json json;
    json["type"] = "PatternDecl";
    json["location"] = serializeSourceLocation(node.location);
    json["name"] = node.name;
    json["parent_pattern"] = node.parent_pattern;
    
    nlohmann::json inputs = nlohmann::json::array();
    for (const auto& input : node.inputs) {
        inputs.push_back(input);
    }
    json["inputs"] = inputs;
    json["body"] = serializeStatements(node.body);
    return json;
}

nlohmann::json ASTSerializer::serializeSignalDecl(const SignalDecl& node) {
    nlohmann::json json;
    json["type"] = "SignalDecl";
    json["location"] = serializeSourceLocation(node.location);
    json["name"] = node.name;
    json["trigger"] = serializeExpression(*node.trigger);
    json["confidence"] = serializeExpression(*node.confidence);
    json["action"] = node.action;
    return json;
}

// Deserialization implementations would continue here...
// For brevity, I'll implement key ones and placeholders for others

std::unique_ptr<NumberLiteral> ASTSerializer::deserializeNumberLiteral(const nlohmann::json& json) {
    auto node = std::make_unique<NumberLiteral>();
    node->location = deserializeSourceLocation(json["location"]);
    node->value = json["value"].get<double>();
    return node;
}

std::unique_ptr<StringLiteral> ASTSerializer::deserializeStringLiteral(const nlohmann::json& json) {
    auto node = std::make_unique<StringLiteral>();
    node->location = deserializeSourceLocation(json["location"]);
    node->value = json["value"].get<std::string>();
    return node;
}

std::unique_ptr<BooleanLiteral> ASTSerializer::deserializeBooleanLiteral(const nlohmann::json& json) {
    auto node = std::make_unique<BooleanLiteral>();
    node->location = deserializeSourceLocation(json["location"]);
    node->value = json["value"].get<bool>();
    return node;
}

std::unique_ptr<Identifier> ASTSerializer::deserializeIdentifier(const nlohmann::json& json) {
    auto node = std::make_unique<Identifier>();
    node->location = deserializeSourceLocation(json["location"]);
    node->name = json["name"].get<std::string>();
    return node;
}

std::unique_ptr<BinaryOp> ASTSerializer::deserializeBinaryOp(const nlohmann::json& json) {
    auto node = std::make_unique<BinaryOp>();
    node->location = deserializeSourceLocation(json["location"]);
    node->left = deserializeExpression(json["left"]);
    node->op = json["op"].get<std::string>();
    node->right = deserializeExpression(json["right"]);
    return node;
}

std::unique_ptr<UnaryOp> ASTSerializer::deserializeUnaryOp(const nlohmann::json& json) {
    auto node = std::make_unique<UnaryOp>();
    node->location = deserializeSourceLocation(json["location"]);
    node->op = json["op"].get<std::string>();
    node->right = deserializeExpression(json["right"]);
    return node;
}

std::unique_ptr<Call> ASTSerializer::deserializeCall(const nlohmann::json& json) {
    auto node = std::make_unique<Call>();
    node->location = deserializeSourceLocation(json["location"]);
    node->callee = json["callee"].get<std::string>();
    node->args = deserializeExpressions(json["args"]);
    return node;
}

std::unique_ptr<MemberAccess> ASTSerializer::deserializeMemberAccess(const nlohmann::json& json) {
    auto node = std::make_unique<MemberAccess>();
    node->location = deserializeSourceLocation(json["location"]);
    node->object = deserializeExpression(json["object"]);
    node->member = json["member"].get<std::string>();
    return node;
}

std::unique_ptr<ArrayLiteral> ASTSerializer::deserializeArrayLiteral(const nlohmann::json& json) {
    auto node = std::make_unique<ArrayLiteral>();
    node->location = deserializeSourceLocation(json["location"]);
    node->elements = deserializeExpressions(json["elements"]);
    return node;
}

std::unique_ptr<ArrayAccess> ASTSerializer::deserializeArrayAccess(const nlohmann::json& json) {
    auto node = std::make_unique<ArrayAccess>();
    node->location = deserializeSourceLocation(json["location"]);
    node->array = deserializeExpression(json["array"]);
    node->index = deserializeExpression(json["index"]);
    return node;
}

std::unique_ptr<WeightedSum> ASTSerializer::deserializeWeightedSum(const nlohmann::json& json) {
    auto node = std::make_unique<WeightedSum>();
    node->location = deserializeSourceLocation(json["location"]);
    
    for (const auto& pairJson : json["pairs"]) {
        auto weight = deserializeExpression(pairJson["weight"]);
        auto value = deserializeExpression(pairJson["value"]);
        node->pairs.emplace_back(std::move(weight), std::move(value));
    }
    return node;
}

std::unique_ptr<AwaitExpression> ASTSerializer::deserializeAwaitExpression(const nlohmann::json& json) {
    auto node = std::make_unique<AwaitExpression>();
    node->location = deserializeSourceLocation(json["location"]);
    node->expression = deserializeExpression(json["expression"]);
    return node;
}

std::unique_ptr<Assignment> ASTSerializer::deserializeAssignment(const nlohmann::json& json) {
    auto node = std::make_unique<Assignment>();
    node->location = deserializeSourceLocation(json["location"]);
    node->name = json["name"].get<std::string>();
    node->type = deserializeTypeAnnotation(json["type_annotation"]);
    node->value = deserializeExpression(json["value"]);
    return node;
}

std::unique_ptr<ExpressionStatement> ASTSerializer::deserializeExpressionStatement(const nlohmann::json& json) {
    auto node = std::make_unique<ExpressionStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    node->expression = deserializeExpression(json["expression"]);
    return node;
}

std::unique_ptr<EvolveStatement> ASTSerializer::deserializeEvolveStatement(const nlohmann::json& json) {
    auto node = std::make_unique<EvolveStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    node->condition = deserializeExpression(json["condition"]);
    node->body = deserializeStatements(json["body"]);
    return node;
}

std::unique_ptr<IfStatement> ASTSerializer::deserializeIfStatement(const nlohmann::json& json) {
    auto node = std::make_unique<IfStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    node->condition = deserializeExpression(json["condition"]);
    node->then_branch = deserializeStatements(json["then_branch"]);
    node->else_branch = deserializeStatements(json["else_branch"]);
    return node;
}

std::unique_ptr<WhileStatement> ASTSerializer::deserializeWhileStatement(const nlohmann::json& json) {
    auto node = std::make_unique<WhileStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    node->condition = deserializeExpression(json["condition"]);
    node->body = deserializeStatements(json["body"]);
    return node;
}

std::unique_ptr<ReturnStatement> ASTSerializer::deserializeReturnStatement(const nlohmann::json& json) {
    auto node = std::make_unique<ReturnStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    if (json.contains("value")) {
        node->value = deserializeExpression(json["value"]);
    }
    return node;
}

std::unique_ptr<FunctionDeclaration> ASTSerializer::deserializeFunctionDeclaration(const nlohmann::json& json) {
    auto node = std::make_unique<FunctionDeclaration>();
    node->location = deserializeSourceLocation(json["location"]);
    node->name = json["name"].get<std::string>();
    node->return_type = deserializeTypeAnnotation(json["return_type"]);
    
    for (const auto& paramJson : json["parameters"]) {
        std::string name = paramJson["name"].get<std::string>();
        TypeAnnotation type = deserializeTypeAnnotation(paramJson["type"]);
        node->parameters.emplace_back(name, type);
    }
    
    node->body = deserializeStatements(json["body"]);
    return node;
}

std::unique_ptr<AsyncFunctionDeclaration> ASTSerializer::deserializeAsyncFunctionDeclaration(const nlohmann::json& json) {
    auto node = std::make_unique<AsyncFunctionDeclaration>();
    node->location = deserializeSourceLocation(json["location"]);
    node->name = json["name"].get<std::string>();
    node->return_type = deserializeTypeAnnotation(json["return_type"]);
    
    for (const auto& paramJson : json["parameters"]) {
        std::string name = paramJson["name"].get<std::string>();
        TypeAnnotation type = deserializeTypeAnnotation(paramJson["type"]);
        node->parameters.emplace_back(name, type);
    }
    
    node->body = deserializeStatements(json["body"]);
    return node;
}

std::unique_ptr<ImportStatement> ASTSerializer::deserializeImportStatement(const nlohmann::json& json) {
    auto node = std::make_unique<ImportStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    node->module_path = json["module_path"].get<std::string>();
    
    for (const auto& import : json["imports"]) {
        node->imports.push_back(import.get<std::string>());
    }
    return node;
}

std::unique_ptr<ExportStatement> ASTSerializer::deserializeExportStatement(const nlohmann::json& json) {
    auto node = std::make_unique<ExportStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    
    for (const auto& export_name : json["exports"]) {
        node->exports.push_back(export_name.get<std::string>());
    }
    return node;
}

std::unique_ptr<TryStatement> ASTSerializer::deserializeTryStatement(const nlohmann::json& json) {
    auto node = std::make_unique<TryStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    node->try_body = deserializeStatements(json["try_body"]);
    node->catch_body = deserializeStatements(json["catch_body"]);
    node->catch_variable = json["catch_variable"].get<std::string>();
    node->finally_body = deserializeStatements(json["finally_body"]);
    return node;
}

std::unique_ptr<ThrowStatement> ASTSerializer::deserializeThrowStatement(const nlohmann::json& json) {
    auto node = std::make_unique<ThrowStatement>();
    node->location = deserializeSourceLocation(json["location"]);
    node->expression = deserializeExpression(json["expression"]);
    return node;
}

std::unique_ptr<StreamDecl> ASTSerializer::deserializeStreamDecl(const nlohmann::json& json) {
    auto node = std::make_unique<StreamDecl>();
    node->location = deserializeSourceLocation(json["location"]);
    node->name = json["name"].get<std::string>();
    node->source = json["source"].get<std::string>();
    
    for (const auto& [key, value] : json["params"].items()) {
        node->params[key] = value.get<std::string>();
    }
    return node;
}

std::unique_ptr<PatternDecl> ASTSerializer::deserializePatternDecl(const nlohmann::json& json) {
    auto node = std::make_unique<PatternDecl>();
    node->location = deserializeSourceLocation(json["location"]);
    node->name = json["name"].get<std::string>();
    node->parent_pattern = json["parent_pattern"].get<std::string>();
    
    for (const auto& input : json["inputs"]) {
        node->inputs.push_back(input.get<std::string>());
    }
    
    node->body = deserializeStatements(json["body"]);
    return node;
}

std::unique_ptr<SignalDecl> ASTSerializer::deserializeSignalDecl(const nlohmann::json& json) {
    auto node = std::make_unique<SignalDecl>();
    node->location = deserializeSourceLocation(json["location"]);
    node->name = json["name"].get<std::string>();
    node->trigger = deserializeExpression(json["trigger"]);
    node->confidence = deserializeExpression(json["confidence"]);
    node->action = json["action"].get<std::string>();
    return node;
}

// Helper methods
nlohmann::json ASTSerializer::serializeSourceLocation(const SourceLocation& location) {
    nlohmann::json json;
    json["line"] = location.line;
    json["column"] = location.column;
    return json;
}

SourceLocation ASTSerializer::deserializeSourceLocation(const nlohmann::json& json) {
    return SourceLocation(json["line"].get<size_t>(), json["column"].get<size_t>());
}

nlohmann::json ASTSerializer::serializeTypeAnnotation(TypeAnnotation type) {
    switch (type) {
        case TypeAnnotation::NUMBER: return "NUMBER";
        case TypeAnnotation::STRING: return "STRING";
        case TypeAnnotation::BOOL: return "BOOL";
        case TypeAnnotation::PATTERN: return "PATTERN";
        case TypeAnnotation::VOID: return "VOID";
        case TypeAnnotation::ARRAY: return "ARRAY";
        case TypeAnnotation::INFERRED: return "INFERRED";
        default: return "INFERRED";
    }
}

TypeAnnotation ASTSerializer::deserializeTypeAnnotation(const nlohmann::json& json) {
    std::string type = json.get<std::string>();
    if (type == "NUMBER") return TypeAnnotation::NUMBER;
    else if (type == "STRING") return TypeAnnotation::STRING;
    else if (type == "BOOL") return TypeAnnotation::BOOL;
    else if (type == "PATTERN") return TypeAnnotation::PATTERN;
    else if (type == "VOID") return TypeAnnotation::VOID;
    else if (type == "ARRAY") return TypeAnnotation::ARRAY;
    else return TypeAnnotation::INFERRED;
}

nlohmann::json ASTSerializer::serializeStatements(const std::vector<std::unique_ptr<Statement>>& statements) {
    nlohmann::json array = nlohmann::json::array();
    for (const auto& stmt : statements) {
        array.push_back(serializeStatement(*stmt));
    }
    return array;
}

std::vector<std::unique_ptr<Statement>> ASTSerializer::deserializeStatements(const nlohmann::json& json) {
    std::vector<std::unique_ptr<Statement>> statements;
    for (const auto& stmtJson : json) {
        statements.push_back(deserializeStatement(stmtJson));
    }
    return statements;
}

nlohmann::json ASTSerializer::serializeExpressions(const std::vector<std::unique_ptr<Expression>>& expressions) {
    nlohmann::json array = nlohmann::json::array();
    for (const auto& expr : expressions) {
        array.push_back(serializeExpression(*expr));
    }
    return array;
}

std::vector<std::unique_ptr<Expression>> ASTSerializer::deserializeExpressions(const nlohmann::json& json) {
    std::vector<std::unique_ptr<Expression>> expressions;
    for (const auto& exprJson : json) {
        expressions.push_back(deserializeExpression(exprJson));
    }
    return expressions;
}

} // namespace dsl::ast
