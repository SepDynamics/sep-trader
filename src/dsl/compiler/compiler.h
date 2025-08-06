#pragma once
#include "ast/nodes.h"
#include "engine/internal/standard_includes.h"
#include <vector>
#include <any>
#include <stdexcept>

namespace dsl::compiler {

// Runtime value types
struct Value {
    enum Type { NUMBER, STRING, BOOLEAN, FUNCTION, PATTERN, STREAM };
    Type type;
    std::any data;
    
    Value() : type(NUMBER), data(0.0) {} // Default constructor
    Value(double d) : type(NUMBER), data(d) {}
    Value(const std::string& s) : type(STRING), data(s) {}
    Value(bool b) : type(BOOLEAN), data(b) {}
    
    template<typename T>
    T get() const { return std::any_cast<T>(data); }
};

// Execution context
struct Context {
    std::unordered_map<std::string, Value> variables;
    std::unordered_map<std::string, std::function<Value(const std::vector<Value>&)>> functions;
    
    void set_variable(const std::string& name, const Value& value) {
        variables[name] = value;
    }
    
    Value get_variable(const std::string& name) const {
        auto it = variables.find(name);
        if (it != variables.end()) {
            return it->second;
        }
        throw std::runtime_error("Undefined variable: " + name);
    }
    
    void set_function(const std::string& name, std::function<Value(const std::vector<Value>&)> func) {
        functions[name] = func;
    }
    
    Value call_function(const std::string& name, const std::vector<Value>& args) const {
        auto it = functions.find(name);
        if (it != functions.end()) {
            return it->second(args);
        }
        throw std::runtime_error("Undefined function: " + name);
    }
};

// Compiled executable
struct CompiledProgram {
    std::function<void(Context&)> execute;
    
    CompiledProgram(std::function<void(Context&)> exec) : execute(exec) {}
};

class Compiler {
private:
    // Compile different AST node types
    std::function<Value(Context&)> compile_expression(const ast::Expression& expr);
    std::function<void(Context&)> compile_statement(const ast::Statement& stmt);
    std::function<void(Context&)> compile_stream_declaration(const ast::StreamDecl& stream);
    std::function<void(Context&)> compile_pattern_declaration(const ast::PatternDecl& pattern);
    std::function<void(Context&)> compile_signal_declaration(const ast::SignalDecl& signal);
    std::function<void(Context&)> compile_memory_declaration(const ast::MemoryDecl& memory);
    
    // Built-in function implementations
    void register_builtin_functions(Context& context);
    Value builtin_qfh(const std::vector<Value>& args);
    Value builtin_qbsa(const std::vector<Value>& args);
    Value builtin_coherence(const std::vector<Value>& args);
    Value builtin_stability(const std::vector<Value>& args);
    Value builtin_entropy(const std::vector<Value>& args);
    Value builtin_weighted_sum(const std::vector<Value>& args);

public:
    CompiledProgram compile(const ast::Program& program);
};

} // namespace dsl::compiler
