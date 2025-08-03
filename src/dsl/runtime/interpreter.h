#pragma once
#include "dsl/ast/nodes.h"
#include <unordered_map>
#include <string>
#include <any>

namespace dsl::runtime {

// A simple variant-like type for our language
using Value = std::any;

// PatternResult holds the variables computed within a pattern
using PatternResult = std::unordered_map<std::string, Value>;

class Environment {
private:
    std::unordered_map<std::string, Value> variables_;
    Environment* enclosing_;

public:
    Environment() : enclosing_(nullptr) {}
    Environment(Environment* enclosing) : enclosing_(enclosing) {}
    
    void define(const std::string& name, const Value& value);
    Value get(const std::string& name);
    void assign(const std::string& name, const Value& value);
    
    // Allow access to variables for pattern result capture
    const std::unordered_map<std::string, Value>& getVariables() const { return variables_; }
};

class Interpreter {
public:
    void interpret(const ast::Program& program);

private:
    Environment globals_;
    Environment* environment_;
    
    // Methods to "visit" and execute each AST node type
    Value evaluate(const ast::Expression& expr);
    void execute(const ast::Statement& stmt);
    
    // Expression evaluation visitors
    Value visit_number_literal(const ast::NumberLiteral& node);
    Value visit_string_literal(const ast::StringLiteral& node);
    Value visit_identifier(const ast::Identifier& node);
    Value visit_binary_op(const ast::BinaryOp& node);
    Value visit_call(const ast::Call& node);
    Value visit_member_access(const ast::MemberAccess& node);
    
    // Statement execution visitors
    void visit_assignment(const ast::Assignment& node);
    void visit_expression_statement(const ast::ExpressionStatement& node);
    
    // Declaration handling
    void execute_stream_decl(const ast::StreamDecl& decl);
    void execute_pattern_decl(const ast::PatternDecl& decl);
    void execute_signal_decl(const ast::SignalDecl& decl);
    
    // Built-in functions
    Value call_builtin_function(const std::string& name, const std::vector<Value>& args);
    
    // Utility functions
    bool is_truthy(const Value& value);
    bool is_equal(const Value& a, const Value& b);
    std::string stringify(const Value& value);
};

} // namespace dsl::runtime
