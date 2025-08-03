#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "../ast/ast_nodes.h"

namespace sep_dsl {
namespace bytecode {

/**
 * Bytecode Instruction Types
 * 
 * Optimized instruction set for DSL execution
 */
enum class OpCode : uint8_t {
    // Stack operations
    LOAD_CONST = 0x01,      // Load constant to stack
    LOAD_VAR = 0x02,        // Load variable to stack
    STORE_VAR = 0x03,       // Store stack top to variable
    
    // Arithmetic operations
    ADD = 0x10,             // a + b
    SUB = 0x11,             // a - b
    MUL = 0x12,             // a * b
    DIV = 0x13,             // a / b
    
    // Comparison operations
    EQ = 0x20,              // a == b
    NE = 0x21,              // a != b
    LT = 0x22,              // a < b
    LE = 0x23,              // a <= b
    GT = 0x24,              // a > b
    GE = 0x25,              // a >= b
    
    // Logical operations
    AND = 0x30,             // a && b
    OR = 0x31,              // a || b
    NOT = 0x32,             // !a
    
    // Control flow
    JUMP = 0x40,            // Unconditional jump
    JUMP_IF_FALSE = 0x41,   // Jump if stack top is false
    JUMP_IF_TRUE = 0x42,    // Jump if stack top is true
    
    // Function calls
    CALL_BUILTIN = 0x50,    // Call built-in function
    CALL_PATTERN = 0x51,    // Call pattern
    
    // Pattern operations
    BEGIN_PATTERN = 0x60,   // Start pattern execution
    END_PATTERN = 0x61,     // End pattern execution
    
    // I/O operations
    PRINT = 0x70,           // Print value
    
    // Special operations
    POP = 0x80,             // Remove top from stack
    DUP = 0x81,             // Duplicate stack top
    HALT = 0xFF             // End execution
};

/**
 * Bytecode Instruction
 */
struct Instruction {
    OpCode opcode;
    std::vector<uint32_t> operands;
    
    Instruction(OpCode op) : opcode(op) {}
    Instruction(OpCode op, uint32_t operand) : opcode(op), operands{operand} {}
    Instruction(OpCode op, std::vector<uint32_t> ops) : opcode(op), operands(std::move(ops)) {}
};

/**
 * Compiled Bytecode Program
 */
class BytecodeProgram {
public:
    std::vector<Instruction> instructions;
    std::vector<std::string> constants;
    std::unordered_map<std::string, uint32_t> pattern_offsets;
    
    void add_instruction(OpCode opcode);
    void add_instruction(OpCode opcode, uint32_t operand);
    void add_instruction(OpCode opcode, std::vector<uint32_t> operands);
    
    uint32_t add_constant(const std::string& value);
    void set_pattern_offset(const std::string& name, uint32_t offset);
    
    size_t size() const { return instructions.size(); }
    void dump() const; // Debug output
};

/**
 * Bytecode Compiler
 * 
 * Compiles AST to optimized bytecode for faster execution
 */
class BytecodeCompiler {
public:
    BytecodeCompiler();
    
    /**
     * Compile AST to bytecode
     */
    std::unique_ptr<BytecodeProgram> compile(const std::vector<std::unique_ptr<ast::ASTNode>>& ast);
    
private:
    std::unique_ptr<BytecodeProgram> program_;
    uint32_t label_counter_;
    
    // Compilation methods
    void compile_node(const ast::ASTNode* node);
    void compile_pattern(const ast::PatternNode* pattern);
    void compile_assignment(const ast::AssignmentNode* assignment);
    void compile_if_statement(const ast::IfNode* if_stmt);
    void compile_print_statement(const ast::PrintNode* print_stmt);
    void compile_expression(const ast::ExpressionNode* expr);
    void compile_binary_op(const ast::BinaryOpNode* binop);
    void compile_function_call(const ast::FunctionCallNode* call);
    void compile_variable_ref(const ast::VariableRefNode* var_ref);
    void compile_literal(const ast::LiteralNode* literal);
    
    // Helper methods
    uint32_t generate_label();
    void patch_jump(size_t instruction_index, uint32_t target);
    
    // Optimization passes
    void optimize_constant_folding();
    void optimize_dead_code_elimination();
    void optimize_jump_threading();
};

/**
 * Bytecode Optimization Statistics
 */
struct OptimizationStats {
    size_t original_instructions = 0;
    size_t optimized_instructions = 0;
    size_t constants_folded = 0;
    size_t dead_code_removed = 0;
    size_t jumps_optimized = 0;
    
    double compression_ratio() const {
        if (original_instructions == 0) return 1.0;
        return static_cast<double>(optimized_instructions) / original_instructions;
    }
};

} // namespace bytecode
} // namespace sep_dsl
