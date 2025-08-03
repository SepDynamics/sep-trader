#pragma once
#include "../ast/nodes.h"
#include <memory>

namespace dsl::optimizer {

class ASTOptimizer {
public:
    // Main optimization entry point
    void optimize(ast::Program& program);
    
private:
    // Optimization passes
    void constant_folding(ast::Program& program);
    void dead_code_elimination(ast::Program& program);
    
    // Expression optimization
    std::unique_ptr<ast::Expression> optimize_expression(std::unique_ptr<ast::Expression> expr);
    std::unique_ptr<ast::BinaryOp> optimize_binary_op(std::unique_ptr<ast::BinaryOp> binary);
    std::unique_ptr<ast::UnaryOp> optimize_unary_op(std::unique_ptr<ast::UnaryOp> unary);
    
    // Statement optimization
    std::unique_ptr<ast::Statement> optimize_statement(std::unique_ptr<ast::Statement> stmt);
    void optimize_pattern(ast::PatternDecl& pattern);
    
    // Helper functions
    bool is_constant_expression(const ast::Expression& expr);
    double evaluate_constant_expression(const ast::Expression& expr);
    std::unique_ptr<ast::NumberLiteral> create_number_literal(double value, const ast::SourceLocation& location);
    
    // Statistics
    int optimizations_applied = 0;
};

} // namespace dsl::optimizer
