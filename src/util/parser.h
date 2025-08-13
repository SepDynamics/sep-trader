#pragma once
#include "nodes.h"
#include "lexer.h"
#include "optimizer.h"

namespace dsl::parser {
class Parser {
public:
    explicit Parser(const std::string& source, bool optimize = true);
    std::unique_ptr<ast::Program> parse();
private:
    // Core parser state
    lexer::Lexer lexer_;
    lexer::Token current_token_;
    bool enable_optimization_;
    void advance();
    void expect(ast::TokenType type, const std::string& message);

    // Declaration parsing
    std::unique_ptr<ast::StreamDecl> parse_stream();
    std::unique_ptr<ast::PatternDecl> parse_pattern();
    std::unique_ptr<ast::SignalDecl> parse_signal();
    
    // Expression parsing
    std::unique_ptr<ast::Expression> parse_expression();
    std::unique_ptr<ast::Expression> parse_logical_or();
    std::unique_ptr<ast::Expression> parse_logical_and();
    std::unique_ptr<ast::Expression> parse_equality();
    std::unique_ptr<ast::Expression> parse_comparison();
    std::unique_ptr<ast::Expression> parse_term();
    std::unique_ptr<ast::Expression> parse_factor();
    std::unique_ptr<ast::Expression> parse_unary();
    std::unique_ptr<ast::Expression> parse_call();
    std::unique_ptr<ast::Expression> parse_primary();
    
    // Statement parsing
    std::unique_ptr<ast::Statement> parse_statement();
    std::vector<std::unique_ptr<ast::Statement>> parse_block();
    
    // Helper methods
    std::vector<std::unique_ptr<ast::Expression>> parse_argument_list();
    std::unordered_map<std::string, std::string> parse_parameter_list();
    
    // Type annotation helpers
    ast::TypeAnnotation parse_type_annotation();
    bool is_type_token(ast::TokenType type);
    
    // Source location helpers
    ast::SourceLocation current_location() const;
    void set_location(ast::Node& node) const;
    
    // Advanced operator precedence table
    enum class Precedence {
        LOWEST = 0,
        TERNARY = 1,         // ? :
        LOGICAL_OR = 2,      // ||
        LOGICAL_AND = 3,     // &&
        BITWISE_OR = 4,      // |
        BITWISE_XOR = 5,     // ^
        BITWISE_AND = 6,     // &
        EQUALITY = 7,        // == !=
        RELATIONAL = 8,      // < <= > >=
        SHIFT = 9,           // << >>
        ADDITIVE = 10,       // + -
        MULTIPLICATIVE = 11, // * / %
        EXPONENTIATION = 12, // **
        UNARY = 13,          // ! - + ~ ++x --x
        POSTFIX = 14,        // x++ x-- [] . ->
        CALL = 15,           // function calls
        MEMBER = 16,         // . ->
        HIGHEST = 17
    };
    
    // Operator precedence table for sophisticated parsing
    struct OperatorInfo {
        Precedence precedence;
        enum class Associativity { LEFT, RIGHT, NONE } associativity;
        bool is_unary;
        bool is_postfix;
    };
    
    const std::unordered_map<ast::TokenType, OperatorInfo>& get_operator_table() const;
    Precedence get_precedence(ast::TokenType token_type) const;
    OperatorInfo::Associativity get_associativity(ast::TokenType token_type) const;
    std::unique_ptr<ast::Expression> parse_precedence(Precedence precedence);
    
    // Advanced constructs
    std::unique_ptr<ast::WeightedSum> parse_weighted_sum();
    std::unique_ptr<ast::EvolveStatement> parse_evolve_statement();
    std::unique_ptr<ast::IfStatement> parse_if_statement();
    std::unique_ptr<ast::WhileStatement> parse_while_statement();
    std::unique_ptr<ast::ForStatement> parse_for_statement();
    std::unique_ptr<ast::FunctionDeclaration> parse_function_declaration();
    std::unique_ptr<ast::ReturnStatement> parse_return_statement();
    std::unique_ptr<ast::ImportStatement> parse_import_statement();
    std::unique_ptr<ast::ExportStatement> parse_export_statement();
    std::unique_ptr<ast::AsyncFunctionDeclaration> parse_async_function_declaration();
    std::unique_ptr<ast::AwaitExpression> parse_await_expression();
    std::unique_ptr<ast::IfExpression> parse_if_expression();
    std::unique_ptr<ast::TryStatement> parse_try_statement();
    std::unique_ptr<ast::ThrowStatement> parse_throw_statement();
};
}
