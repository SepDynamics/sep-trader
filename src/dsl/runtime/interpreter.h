#pragma once
#include <any>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/ast/nodes.h"

namespace dsl::runtime {

    // Forward declarations
    class Environment;
    class Interpreter;
    class Callable;

    // A simple variant-like type for our language
    using Value = std::any;

    // PatternResult holds the variables computed within a pattern
    using PatternResult = std::unordered_map<std::string, Value>;

    // Exception for handling return statements
    class ReturnException : public std::runtime_error
    {
    public:
        ReturnException(const Value& value) : std::runtime_error("return"), value_(value) {}
        const Value& value() const { return value_; }

    private:
        Value value_;
    };

    // Interface for callable objects
    class Callable
    {
    public:
        virtual ~Callable() = default;
        virtual Value call(Interpreter& interpreter, const std::vector<Value>& arguments) = 0;
    };

    // User-defined function implementation
    class UserFunction : public Callable
    {
    public:
        UserFunction(const ast::FunctionDeclaration& declaration, Environment* closure)
            : declaration_(declaration), closure_(closure)
        {
        }

        Value call(Interpreter& interpreter, const std::vector<Value>& arguments) override;

    private:
        const ast::FunctionDeclaration& declaration_;
        Environment* closure_;
    };

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
    friend class UserFunction;

public:
    Interpreter();
    void interpret(const ast::Program& program);
    
    // Variable access
    Value get_global_variable(const std::string& name);
    bool has_global_variable(const std::string& name);
    const std::unordered_map<std::string, Value>& get_global_variables() const;

private:
    Environment globals_;
    Environment* environment_;
    const ast::Program* program_;  // Pointer to the current program being interpreted
    std::unordered_map<std::string, std::function<Value(const std::vector<Value>&)>> builtins_;

    // Methods to "visit" and execute each AST node type
    Value evaluate(const ast::Expression& expr);
    void execute(const ast::Statement& stmt);
    
    // Expression evaluation visitors
    Value visit_number_literal(const ast::NumberLiteral& node);
    Value visit_string_literal(const ast::StringLiteral& node);
    Value visit_boolean_literal(const ast::BooleanLiteral& node);
    Value visit_identifier(const ast::Identifier& node);
    Value visit_binary_op(const ast::BinaryOp& node);
    Value visit_unary_op(const ast::UnaryOp& node);
    Value visit_call(const ast::Call& node);
    Value visit_member_access(const ast::MemberAccess& node);
    Value visit_weighted_sum(const ast::WeightedSum& node);

    // Statement execution visitors
    void visit_assignment(const ast::Assignment& node);
    void visit_expression_statement(const ast::ExpressionStatement& node);
    void visit_evolve_statement(const ast::EvolveStatement& node);
    void visit_if_statement(const ast::IfStatement& node);
    void visit_while_statement(const ast::WhileStatement& node);
    void visit_function_declaration(const ast::FunctionDeclaration& node);
    void visit_return_statement(const ast::ReturnStatement& node);

    // Declaration handling
    void execute_stream_decl(const ast::StreamDecl& decl);
    void execute_pattern_decl(const ast::PatternDecl& decl);
    void execute_signal_decl(const ast::SignalDecl& decl);
    
    // Built-in functions
    void register_builtins();
    Value call_builtin_function(const std::string& name, const std::vector<Value>& args);
    
    // Utility functions
    bool is_truthy(const Value& value);
    bool is_equal(const Value& a, const Value& b);
    std::string stringify(const Value& value);
};

} // namespace dsl::runtime
