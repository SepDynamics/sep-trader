#pragma once

#include "compiler.h"
#include <stack>
#include <unordered_map>
#include <functional>
#include <chrono>

namespace sep_dsl {
namespace bytecode {

/**
 * Runtime Value Types for Bytecode VM
 */
enum class ValueType : uint8_t {
    NUMBER,
    STRING,
    BOOLEAN,
    NIL
};

/**
 * Runtime Value
 */
struct Value {
    ValueType type;
    union {
        double number;
        bool boolean;
    };
    std::string string_value; // For strings
    
    Value() : type(ValueType::NIL) {}
    Value(double n) : type(ValueType::NUMBER), number(n) {}
    Value(bool b) : type(ValueType::BOOLEAN), boolean(b) {}
    Value(const std::string& s) : type(ValueType::STRING), string_value(s) {}
    
    std::string to_string() const;
    bool is_truthy() const;
    bool equals(const Value& other) const;
};

/**
 * Virtual Machine Execution Context
 */
class VMExecutionContext {
public:
    std::unordered_map<std::string, Value> variables;
    std::unordered_map<std::string, std::unordered_map<std::string, Value>> pattern_variables;
    std::string current_pattern;
    
    void set_variable(const std::string& name, const Value& value);
    Value get_variable(const std::string& name) const;
    void enter_pattern(const std::string& name);
    void exit_pattern();
    void set_pattern_variable(const std::string& var_name, const Value& value);
    Value get_pattern_variable(const std::string& pattern_name, const std::string& var_name) const;
};

/**
 * Virtual Machine Performance Metrics
 */
struct VMMetrics {
    size_t instructions_executed = 0;
    size_t function_calls = 0;
    size_t pattern_executions = 0;
    std::chrono::duration<double> execution_time{0};
    std::chrono::duration<double> builtin_time{0};
    
    void reset();
    void dump() const;
};

/**
 * Bytecode Virtual Machine
 * 
 * High-performance execution engine for compiled DSL bytecode
 */
class BytecodeVM {
public:
    BytecodeVM();
    ~BytecodeVM();
    
    /**
     * Execute compiled bytecode program
     */
    void execute(const BytecodeProgram& program);
    
    /**
     * Register built-in function
     */
    void register_builtin(const std::string& name, 
                         std::function<Value(const std::vector<Value>&)> func);
    
    /**
     * Get variable value from execution context
     */
    Value get_variable(const std::string& name) const;
    
    /**
     * Get pattern variable with dot notation support
     */
    Value get_pattern_variable(const std::string& full_name) const;
    
    /**
     * Get execution metrics
     */
    const VMMetrics& get_metrics() const { return metrics_; }
    
    /**
     * Reset VM state
     */
    void reset();
    
    /**
     * Enable/disable debug mode
     */
    void set_debug_mode(bool enabled) { debug_mode_ = enabled; }
    
private:
    // Execution state
    std::stack<Value> stack_;
    VMExecutionContext context_;
    const BytecodeProgram* program_;
    size_t pc_; // Program counter
    
    // Built-in functions
    std::unordered_map<std::string, std::function<Value(const std::vector<Value>&)>> builtins_;
    
    // Performance tracking
    VMMetrics metrics_;
    bool debug_mode_;
    
    // Execution methods
    void execute_instruction(const Instruction& instr);
    void op_load_const(uint32_t const_index);
    void op_load_var(uint32_t const_index);
    void op_store_var(uint32_t const_index);
    void op_add();
    void op_sub();
    void op_mul();
    void op_div();
    void op_eq();
    void op_ne();
    void op_lt();
    void op_le();
    void op_gt();
    void op_ge();
    void op_and();
    void op_or();
    void op_not();
    void op_jump(uint32_t target);
    void op_jump_if_false(uint32_t target);
    void op_jump_if_true(uint32_t target);
    void op_call_builtin(uint32_t func_index, uint32_t arg_count);
    void op_call_pattern(uint32_t pattern_index);
    void op_begin_pattern(uint32_t name_index);
    void op_end_pattern();
    void op_print(uint32_t arg_count);
    void op_pop();
    void op_dup();
    
    // Helper methods
    Value pop_stack();
    void push_stack(const Value& value);
    Value peek_stack() const;
    void debug_print_instruction(const Instruction& instr) const;
    void debug_print_stack() const;
    
    // Built-in function implementations
    void register_default_builtins();
    Value builtin_measure_coherence(const std::vector<Value>& args);
    Value builtin_measure_entropy(const std::vector<Value>& args);
    Value builtin_extract_bits(const std::vector<Value>& args);
    Value builtin_qfh_analyze(const std::vector<Value>& args);
    Value builtin_manifold_optimize(const std::vector<Value>& args);
    
    // Error handling
    void runtime_error(const std::string& message);
    void type_error(const std::string& expected, const Value& got);
    void arity_error(const std::string& func_name, size_t expected, size_t got);
};

/**
 * Bytecode Execution Exception
 */
class BytecodeExecutionError : public std::runtime_error {
public:
    BytecodeExecutionError(const std::string& message, size_t pc)
        : std::runtime_error(message), pc_(pc) {}
    
    size_t program_counter() const { return pc_; }
    
private:
    size_t pc_;
};

} // namespace bytecode
} // namespace sep_dsl
