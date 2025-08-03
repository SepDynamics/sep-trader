#include "optimizer.h"
#include <iostream>
#include <cmath>

namespace dsl::optimizer {

void ASTOptimizer::optimize(ast::Program& program) {
    optimizations_applied = 0;
    
    // Apply optimization passes
    constant_folding(program);
    dead_code_elimination(program);
    
    std::cout << "AST Optimizer: Applied " << optimizations_applied << " optimizations" << std::endl;
}

void ASTOptimizer::constant_folding(ast::Program& program) {
    // Optimize patterns
    for (auto& pattern : program.patterns) {
        optimize_pattern(*pattern);
    }
}

void ASTOptimizer::dead_code_elimination(ast::Program& program) {
    // Simple dead code elimination - remove unreachable statements after return
    for (auto& pattern : program.patterns) {
        for (size_t i = 0; i < pattern->body.size(); i++) {
            if (auto return_stmt = dynamic_cast<ast::ReturnStatement*>(pattern->body[i].get())) {
                // Remove any statements after return
                if (i + 1 < pattern->body.size()) {
                    pattern->body.erase(pattern->body.begin() + i + 1, pattern->body.end());
                    optimizations_applied++;
                    std::cout << "  Removed dead code after return statement" << std::endl;
                }
                break;
            }
        }
    }
}

void ASTOptimizer::optimize_pattern(ast::PatternDecl& pattern) {
    // Optimize each statement in the pattern
    for (auto& stmt : pattern.body) {
        stmt = optimize_statement(std::move(stmt));
    }
}

std::unique_ptr<ast::Statement> ASTOptimizer::optimize_statement(std::unique_ptr<ast::Statement> stmt) {
    // Optimize assignment statements
    if (auto assignment = dynamic_cast<ast::Assignment*>(stmt.get())) {
        assignment->value = optimize_expression(std::move(assignment->value));
    }
    
    // Optimize expression statements
    if (auto expr_stmt = dynamic_cast<ast::ExpressionStatement*>(stmt.get())) {
        expr_stmt->expression = optimize_expression(std::move(expr_stmt->expression));
    }
    
    // Optimize function declarations
    if (auto func_decl = dynamic_cast<ast::FunctionDeclaration*>(stmt.get())) {
        for (auto& body_stmt : func_decl->body) {
            body_stmt = optimize_statement(std::move(body_stmt));
        }
    }
    
    // Optimize return statements
    if (auto return_stmt = dynamic_cast<ast::ReturnStatement*>(stmt.get())) {
        if (return_stmt->value) {
            return_stmt->value = optimize_expression(std::move(return_stmt->value));
        }
    }
    
    return stmt;
}

std::unique_ptr<ast::Expression> ASTOptimizer::optimize_expression(std::unique_ptr<ast::Expression> expr) {
    // Optimize binary operations
    if (auto binary = dynamic_cast<ast::BinaryOp*>(expr.get())) {
        auto binary_ptr = std::unique_ptr<ast::BinaryOp>(static_cast<ast::BinaryOp*>(expr.release()));
        return optimize_binary_op(std::move(binary_ptr));
    }
    
    // Optimize unary operations
    if (auto unary = dynamic_cast<ast::UnaryOp*>(expr.get())) {
        auto unary_ptr = std::unique_ptr<ast::UnaryOp>(static_cast<ast::UnaryOp*>(expr.release()));
        return optimize_unary_op(std::move(unary_ptr));
    }
    
    return expr;
}

std::unique_ptr<ast::BinaryOp> ASTOptimizer::optimize_binary_op(std::unique_ptr<ast::BinaryOp> binary) {
    // First optimize operands
    binary->left = optimize_expression(std::move(binary->left));
    binary->right = optimize_expression(std::move(binary->right));
    
    // Check if both operands are constants
    if (is_constant_expression(*binary->left) && is_constant_expression(*binary->right)) {
        double left_val = evaluate_constant_expression(*binary->left);
        double right_val = evaluate_constant_expression(*binary->right);
        double result = 0.0;
        
        // Evaluate the operation
        if (binary->op == "+") {
            result = left_val + right_val;
        } else if (binary->op == "-") {
            result = left_val - right_val;
        } else if (binary->op == "*") {
            result = left_val * right_val;
        } else if (binary->op == "/") {
            if (right_val != 0.0) {
                result = left_val / right_val;
            } else {
                return binary; // Don't optimize division by zero
            }
        } else {
            return binary; // Unknown operator
        }
        
        optimizations_applied++;
        std::cout << "  Constant folding: " << left_val << " " << binary->op << " " << right_val << " = " << result << std::endl;
        
        // Return a constant literal
        return std::unique_ptr<ast::BinaryOp>(reinterpret_cast<ast::BinaryOp*>(
            create_number_literal(result, binary->location).release()
        ));
    }
    
    return binary;
}

std::unique_ptr<ast::UnaryOp> ASTOptimizer::optimize_unary_op(std::unique_ptr<ast::UnaryOp> unary) {
    // First optimize operand
    unary->right = optimize_expression(std::move(unary->right));
    
    // Check if operand is constant
    if (is_constant_expression(*unary->right)) {
        double operand_val = evaluate_constant_expression(*unary->right);
        double result = 0.0;
        
        if (unary->op == "-") {
            result = -operand_val;
        } else if (unary->op == "!") {
            result = operand_val == 0.0 ? 1.0 : 0.0;
        } else {
            return unary; // Unknown operator
        }
        
        optimizations_applied++;
        std::cout << "  Constant folding: " << unary->op << operand_val << " = " << result << std::endl;
        
        // Return a constant literal
        return std::unique_ptr<ast::UnaryOp>(reinterpret_cast<ast::UnaryOp*>(
            create_number_literal(result, unary->location).release()
        ));
    }
    
    return unary;
}

bool ASTOptimizer::is_constant_expression(const ast::Expression& expr) {
    // Check if expression is a number literal
    if (dynamic_cast<const ast::NumberLiteral*>(&expr)) {
        return true;
    }
    
    // Check if expression is a boolean literal
    if (dynamic_cast<const ast::BooleanLiteral*>(&expr)) {
        return true;
    }
    
    // Binary operations are constant if both operands are constant
    if (auto binary = dynamic_cast<const ast::BinaryOp*>(&expr)) {
        return is_constant_expression(*binary->left) && is_constant_expression(*binary->right);
    }
    
    // Unary operations are constant if operand is constant
    if (auto unary = dynamic_cast<const ast::UnaryOp*>(&expr)) {
        return is_constant_expression(*unary->right);
    }
    
    return false;
}

double ASTOptimizer::evaluate_constant_expression(const ast::Expression& expr) {
    // Evaluate number literals
    if (auto number = dynamic_cast<const ast::NumberLiteral*>(&expr)) {
        return number->value;
    }
    
    // Evaluate boolean literals
    if (auto boolean = dynamic_cast<const ast::BooleanLiteral*>(&expr)) {
        return boolean->value ? 1.0 : 0.0;
    }
    
    // Evaluate binary operations
    if (auto binary = dynamic_cast<const ast::BinaryOp*>(&expr)) {
        double left = evaluate_constant_expression(*binary->left);
        double right = evaluate_constant_expression(*binary->right);
        
        if (binary->op == "+") return left + right;
        if (binary->op == "-") return left - right;
        if (binary->op == "*") return left * right;
        if (binary->op == "/") return right != 0.0 ? left / right : 0.0;
    }
    
    // Evaluate unary operations
    if (auto unary = dynamic_cast<const ast::UnaryOp*>(&expr)) {
        double operand = evaluate_constant_expression(*unary->right);
        
        if (unary->op == "-") return -operand;
        if (unary->op == "!") return operand == 0.0 ? 1.0 : 0.0;
    }
    
    return 0.0; // Default fallback
}

std::unique_ptr<ast::NumberLiteral> ASTOptimizer::create_number_literal(double value, const ast::SourceLocation& location) {
    auto literal = std::make_unique<ast::NumberLiteral>();
    literal->value = value;
    literal->location = location;
    return literal;
}

} // namespace dsl::optimizer
