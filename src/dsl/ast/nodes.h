#pragma once

#include <string>
#include <vector>
#include <memory>
#include <variant>

namespace sep::dsl::ast {

    // Forward declarations
    struct Expression;
    
    // Statement placeholder (can be removed when not needed)
    struct Statement {
        // TODO: Define statement types as needed
    };

    // Basic types
    enum class OperationType {
        QFH_ANALYZE,
        QBSA_ANALYZE,
        MANIFOLD_OPTIMIZE,
        MEASURE_COHERENCE,
        MEASURE_STABILITY,
        MEASURE_ENTROPY,
        WEIGHTED_SUM,
        EXTRACT_BITS,
        DETECT_COLLAPSE
    };

    enum class TriggerType {
        THRESHOLD,
        PATTERN_MATCH,
        TIME_BASED,
        CONDITIONAL
    };

    enum class MemoryTier {
        STM,  // Short-term memory
        MTM,  // Medium-term memory  
        LTM   // Long-term memory
    };

    // Expression nodes
    struct LiteralExpression {
        std::variant<double, int, std::string, bool> value;
    };

    struct VariableExpression {
        std::string name;
    };

    struct BinaryExpression {
        std::unique_ptr<Expression> left;
        std::string operator_;
        std::unique_ptr<Expression> right;
    };

    struct FunctionCallExpression {
        std::string function_name;
        std::vector<std::unique_ptr<Expression>> arguments;
    };

    struct Expression {
        std::variant<
            LiteralExpression,
            VariableExpression,
            BinaryExpression,
            FunctionCallExpression
        > data;
    };

    // Input source definitions
    struct InputNode {
        std::string name;
        std::string source_type; // "stream", "file", "memory", etc.
        std::string source_path;
        std::vector<std::pair<std::string, std::unique_ptr<Expression>>> parameters;
    };

    // Computation nodes
    struct ComputationNode {
        std::string result_variable;
        OperationType operation;
        std::vector<std::unique_ptr<Expression>> arguments;
        std::vector<std::pair<std::string, std::unique_ptr<Expression>>> parameters;
    };

    // Evolution rules
    struct EvolutionRule {
        std::unique_ptr<Expression> condition;
        std::vector<std::unique_ptr<Statement>> actions;
    };

    // Memory management rules
    struct MemoryRule {
        std::string pattern_reference;
        MemoryTier target_tier;
        std::unique_ptr<Expression> condition;
    };

    // Signal definitions
    struct SignalNode {
        std::string name;
        std::unique_ptr<Expression> trigger_condition;
        std::unique_ptr<Expression> confidence_expression;
        std::string action; // BUY, SELL, HOLD, etc.
    };

    // Pattern definitions (main computational units)
    struct PatternNode {
        std::string name;
        std::vector<InputNode> inputs;
        std::vector<ComputationNode> computations;
        std::vector<EvolutionRule> evolution_rules;
        std::vector<MemoryRule> memory_rules;
    };

    // Top-level program structure
    struct Program {
        std::vector<InputNode> global_inputs;
        std::vector<PatternNode> patterns;
        std::vector<SignalNode> signals;
        std::vector<MemoryRule> global_memory_rules;
    };

    // Utility functions for AST manipulation
    namespace util {
        std::string expressionToString(const Expression& expr);
        std::vector<std::string> extractVariables(const Expression& expr);
        bool isConstantExpression(const Expression& expr);
    }

} // namespace sep::dsl::ast
