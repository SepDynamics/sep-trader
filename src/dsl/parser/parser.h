#pragma once
#include "dsl/ast/nodes.h"
#include "dsl/lexer/lexer.h"

namespace dsl::parser {
class Parser {
public:
    explicit Parser(const std::string& source);
    std::unique_ptr<ast::Program> parse();
private:
    // Core parser state
    lexer::Lexer lexer_;
    lexer::Token current_token_;
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
    
    // Advanced constructs
    std::unique_ptr<ast::WeightedSum> parse_weighted_sum();
    std::unique_ptr<ast::EvolveStatement> parse_evolve_statement();
};
}
