#pragma once

#ifndef SRC_UTIL_COMPILER_MERGED_H
#define SRC_UTIL_COMPILER_MERGED_H

#include <variant>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

// Forward declarations for AST types
namespace dsl {
namespace ast {
    struct Expression;
    struct Statement;
    struct StreamDecl;
    struct PatternDecl;
    struct SignalDecl;
    struct MemoryDecl;
    struct Program;
}
}

// DSL compiler types
namespace dsl {
namespace compiler {

// Forward declarations
class Context;

// Value type for DSL operations
using Value = std::variant<
    std::monostate,  // null/undefined
    double,          // numbers
    std::string,     // strings
    bool             // booleans
>;

// Bytecode instruction opcodes
enum class OpCode : uint8_t {
    LOAD_CONST,
    LOAD_VAR,
    STORE_VAR,
    ADD,
    SUB,
    MUL,
    DIV,
    CALL,
    RETURN,
    JUMP,
    JUMP_IF_FALSE,
    POP,
    HALT
};

// Bytecode instruction
struct Instruction {
    OpCode opcode;
    uint32_t operand;  // Generic operand (const index, var index, etc.)
    
    Instruction(OpCode op = OpCode::HALT, uint32_t arg = 0)
        : opcode(op), operand(arg) {}
};

// Bytecode program container
class BytecodeProgram {
public:
    std::vector<Instruction> instructions;
    std::vector<Value> constants;
    std::vector<std::string> variable_names;
    
    BytecodeProgram() = default;
    
    void add_instruction(OpCode opcode, uint32_t operand = 0) {
        instructions.emplace_back(opcode, operand);
    }
    
    uint32_t add_constant(const Value& value) {
        constants.push_back(value);
        return static_cast<uint32_t>(constants.size() - 1);
    }
    
    uint32_t add_variable(const std::string& name) {
        variable_names.push_back(name);
        return static_cast<uint32_t>(variable_names.size() - 1);
    }
    
    size_t size() const { return instructions.size(); }
    bool empty() const { return instructions.empty(); }
};

// Compiled program wrapper
class CompiledProgram {
private:
    std::function<void(Context&)> executor_;

public:
    CompiledProgram(std::function<void(Context&)> exec) : executor_(exec) {}
    
    void execute(Context& context) {
        if (executor_) {
            executor_(context);
        }
    }
};

// Forward declarations for AST types (simplified stubs)
namespace ast {
    struct StreamDecl { std::string name; std::string source; };
    struct PatternDecl { std::string name; std::string body; };
    struct SignalDecl {
        std::string name;
        std::string trigger;
        std::string confidence;
        std::string action;
    };
    struct MemoryDecl { std::string name; std::vector<std::string> rules; };
    struct Statement { std::string content; };
    struct AssignmentStmt { std::string variable; std::string expression; };
    
    struct Program {
        std::vector<std::shared_ptr<StreamDecl>> streams;
        std::vector<std::shared_ptr<PatternDecl>> patterns;
        std::vector<std::shared_ptr<SignalDecl>> signals;
        std::shared_ptr<MemoryDecl> memory;
        std::vector<std::shared_ptr<Statement>> statements;
    };
}

// DSL Compiler class
class Compiler {
public:
    Compiler() = default;
    
    CompiledProgram compile(const ast::Program& program);
    
    std::function<void(Context&)> compile_stream_declaration(const dsl::compiler::ast::StreamDecl& stream);
    std::function<void(Context&)> compile_pattern_declaration(const dsl::compiler::ast::PatternDecl& pattern);
    std::function<void(Context&)> compile_signal_declaration(const dsl::compiler::ast::SignalDecl& signal);
    std::function<void(Context&)> compile_memory_declaration(const dsl::compiler::ast::MemoryDecl& memory);
    std::function<void(Context&)> compile_statement(const dsl::compiler::ast::Statement& stmt);
    std::function<Value(Context&)> compile_expression(const dsl::ast::Expression& expr);
    
    void register_builtin_functions(Context& context);
};

// Context for DSL execution
class Context {
public:
    Context() = default;
    
    // Variable management
    void set_variable(const std::string& name, const Value& value) {
        variables_[name] = value;
    }
    
    Value get_variable(const std::string& name) const {
        auto it = variables_.find(name);
        return (it != variables_.end()) ? it->second : Value{};
    }
    
    // Function registration
    void register_function(const std::string& name,
                          std::function<Value(const std::vector<Value>&)> func) {
        functions_[name] = func;
    }
    
    // Alias for compatibility
    void set_function(const std::string& name,
                     std::function<Value(const std::vector<Value>&)> func) {
        register_function(name, func);
    }
    
    std::function<Value(const std::vector<Value>&)> get_function(const std::string& name) const {
        auto it = functions_.find(name);
        return (it != functions_.end()) ? it->second : nullptr;
    }

private:
    std::unordered_map<std::string, Value> variables_;
    std::unordered_map<std::string, std::function<Value(const std::vector<Value>&)>> functions_;
};


} // namespace compiler
} // namespace dsl

#ifndef SRC_UTIL_BYTECODE_COMPILER_H
#define SRC_UTIL_BYTECODE_COMPILER_H
// Content from src/util/bytecode/compiler.h
namespace sep {
namespace util {
namespace bytecode {
class Compiler {};
}
}
}
#endif

#ifndef SRC_UTIL_COMPILER_COMPILER_H
#define SRC_UTIL_COMPILER_COMPILER_H
// Content from src/util/compiler/compiler.h
namespace sep {
namespace util {
namespace compiler {
class Compiler {};
}
}
}
#endif

#endif // SRC_UTIL_COMPILER_MERGED_H
