#include "parser.h"
#include <stdexcept>

namespace dsl::parser {

Parser::Parser(const std::string& source) : lexer_(source), current_token_(ast::TokenType::EOF_TOKEN, "", 0, 0) {
    advance();
}

void Parser::advance() {
    current_token_ = lexer_.next_token();
}

void Parser::expect(ast::TokenType type, const std::string& message) {
    if (current_token_.type == type) {
        advance();
    } else {
        throw std::runtime_error(message + " at line " + std::to_string(current_token_.line));
    }
}

std::unique_ptr<ast::Program> Parser::parse() {
    auto program = std::make_unique<ast::Program>();
    
    while (current_token_.type != ast::TokenType::EOF_TOKEN) {
        if (current_token_.type == ast::TokenType::STREAM) {
            program->streams.push_back(parse_stream());
        } else if (current_token_.type == ast::TokenType::PATTERN) {
            program->patterns.push_back(parse_pattern());
        } else if (current_token_.type == ast::TokenType::SIGNAL) {
            program->signals.push_back(parse_signal());
        } else {
            throw std::runtime_error("Unexpected token: " + current_token_.value + " at line " + std::to_string(current_token_.line));
        }
    }
    
    return program;
}

std::unique_ptr<ast::StreamDecl> Parser::parse_stream() {
    expect(ast::TokenType::STREAM, "Expected 'stream' keyword.");
    
    auto name = current_token_.value;
    expect(ast::TokenType::IDENTIFIER, "Expected stream name.");
    
    expect(ast::TokenType::FROM, "Expected 'from' keyword.");
    
    auto source = current_token_.value;
    expect(ast::TokenType::STRING, "Expected stream source.");
    
    auto stream_decl = std::make_unique<ast::StreamDecl>();
    stream_decl->name = name;
    stream_decl->source = source;
    
    // Optional parameters
    if (current_token_.type == ast::TokenType::LBRACE) {
        advance(); // consume '{'
        stream_decl->params = parse_parameter_list();
        expect(ast::TokenType::RBRACE, "Expected '}'");
    }
    
    return stream_decl;
}

std::unique_ptr<ast::PatternDecl> Parser::parse_pattern() {
    expect(ast::TokenType::PATTERN, "Expected 'pattern' keyword.");
    
    auto name = current_token_.value;
    expect(ast::TokenType::IDENTIFIER, "Expected pattern name.");
    
    auto pattern_decl = std::make_unique<ast::PatternDecl>();
    pattern_decl->name = name;
    
    expect(ast::TokenType::LBRACE, "Expected '{'");
    
    // Parse input declarations
    while (current_token_.type == ast::TokenType::INPUT) {
        advance(); // consume 'input'
        expect(ast::TokenType::COLON, "Expected ':'");
        
        auto input_name = current_token_.value;
        expect(ast::TokenType::IDENTIFIER, "Expected input name");
        
        pattern_decl->inputs.push_back(input_name);
    }
    
    // Parse pattern body (statements)
    pattern_decl->body = parse_block();
    
    expect(ast::TokenType::RBRACE, "Expected '}'");
    
    return pattern_decl;
}

std::unique_ptr<ast::SignalDecl> Parser::parse_signal() {
    expect(ast::TokenType::SIGNAL, "Expected 'signal' keyword.");
    
    auto name = current_token_.value;
    expect(ast::TokenType::IDENTIFIER, "Expected signal name.");
    
    auto signal_decl = std::make_unique<ast::SignalDecl>();
    signal_decl->name = name;
    
    expect(ast::TokenType::LBRACE, "Expected '{'");
    
    // Parse signal properties
    while (current_token_.type != ast::TokenType::RBRACE && current_token_.type != ast::TokenType::EOF_TOKEN) {
        if (current_token_.value == "trigger") {
            advance(); // consume 'trigger'
            expect(ast::TokenType::COLON, "Expected ':'");
            signal_decl->trigger = parse_expression();
        } else if (current_token_.value == "confidence") {
            advance(); // consume 'confidence'
            expect(ast::TokenType::COLON, "Expected ':'");
            signal_decl->confidence = parse_expression();
        } else if (current_token_.value == "action") {
            advance(); // consume 'action'
            expect(ast::TokenType::COLON, "Expected ':'");
            signal_decl->action = current_token_.value;
            expect(ast::TokenType::IDENTIFIER, "Expected action name");
        } else {
            advance(); // Skip unknown properties
        }
    }
    
    expect(ast::TokenType::RBRACE, "Expected '}'");
    
    return signal_decl;
}

std::vector<std::unique_ptr<ast::Statement>> Parser::parse_block() {
    std::vector<std::unique_ptr<ast::Statement>> statements;
    
    while (current_token_.type != ast::TokenType::RBRACE && current_token_.type != ast::TokenType::EOF_TOKEN) {
        statements.push_back(parse_statement());
    }
    
    return statements;
}

std::unique_ptr<ast::Statement> Parser::parse_statement() {
    // Check for assignment (identifier = expression)
    if (current_token_.type == ast::TokenType::IDENTIFIER) {
        // Look ahead to see if this is an assignment
        lexer::Token next = lexer_.peek_token();
        if (next.type == ast::TokenType::ASSIGN) {
            auto name = current_token_.value;
            advance(); // consume identifier
            expect(ast::TokenType::ASSIGN, "Expected '='");
            auto value = parse_expression();
            
            auto assignment = std::make_unique<ast::Assignment>();
            assignment->name = name;
            assignment->value = std::move(value);
            return assignment;
        }
    }
    
    // Otherwise, it's an expression statement
    auto expr = parse_expression();
    auto expr_stmt = std::make_unique<ast::ExpressionStatement>();
    expr_stmt->expression = std::move(expr);
    return expr_stmt;
}

std::unique_ptr<ast::Expression> Parser::parse_expression() {
    return parse_logical_or();
}

std::unique_ptr<ast::Expression> Parser::parse_logical_or() {
    auto expr = parse_logical_and();
    
    while (current_token_.type == ast::TokenType::OR) {
        auto op = current_token_.value;
        advance();
        auto right = parse_logical_and();
        
        auto binary_op = std::make_unique<ast::BinaryOp>();
        binary_op->left = std::move(expr);
        binary_op->op = op;
        binary_op->right = std::move(right);
        expr = std::move(binary_op);
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_logical_and() {
    auto expr = parse_equality();
    
    while (current_token_.type == ast::TokenType::AND) {
        auto op = current_token_.value;
        advance();
        auto right = parse_equality();
        
        auto binary_op = std::make_unique<ast::BinaryOp>();
        binary_op->left = std::move(expr);
        binary_op->op = op;
        binary_op->right = std::move(right);
        expr = std::move(binary_op);
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_equality() {
    auto expr = parse_comparison();
    
    while (current_token_.type == ast::TokenType::EQ || current_token_.type == ast::TokenType::NE) {
        auto op = current_token_.value;
        advance();
        auto right = parse_comparison();
        
        auto binary_op = std::make_unique<ast::BinaryOp>();
        binary_op->left = std::move(expr);
        binary_op->op = op;
        binary_op->right = std::move(right);
        expr = std::move(binary_op);
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_comparison() {
    auto expr = parse_term();
    
    while (current_token_.type == ast::TokenType::GT || current_token_.type == ast::TokenType::GE ||
           current_token_.type == ast::TokenType::LT || current_token_.type == ast::TokenType::LE) {
        auto op = current_token_.value;
        advance();
        auto right = parse_term();
        
        auto binary_op = std::make_unique<ast::BinaryOp>();
        binary_op->left = std::move(expr);
        binary_op->op = op;
        binary_op->right = std::move(right);
        expr = std::move(binary_op);
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_term() {
    auto expr = parse_factor();
    
    while (current_token_.type == ast::TokenType::PLUS || current_token_.type == ast::TokenType::MINUS) {
        auto op = current_token_.value;
        advance();
        auto right = parse_factor();
        
        auto binary_op = std::make_unique<ast::BinaryOp>();
        binary_op->left = std::move(expr);
        binary_op->op = op;
        binary_op->right = std::move(right);
        expr = std::move(binary_op);
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_factor() {
    auto expr = parse_unary();
    
    while (current_token_.type == ast::TokenType::MULTIPLY || current_token_.type == ast::TokenType::DIVIDE) {
        auto op = current_token_.value;
        advance();
        auto right = parse_unary();
        
        auto binary_op = std::make_unique<ast::BinaryOp>();
        binary_op->left = std::move(expr);
        binary_op->op = op;
        binary_op->right = std::move(right);
        expr = std::move(binary_op);
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_unary() {
    if (current_token_.type == ast::TokenType::NOT || current_token_.type == ast::TokenType::MINUS) {
        // TODO: Add UnaryOp node type for proper unary expression handling
        advance();
        return parse_unary();
    }
    
    return parse_call();
}

std::unique_ptr<ast::Expression> Parser::parse_call() {
    auto expr = parse_primary();
    
    while (true) {
        if (current_token_.type == ast::TokenType::LPAREN) {
            // This is a function call - expr should be an Identifier
            advance(); // consume '('
            
            auto call = std::make_unique<ast::Call>();
            // Extract function name from identifier
            if (auto identifier = dynamic_cast<ast::Identifier*>(expr.get())) {
                call->callee = identifier->name;
            } else {
                throw std::runtime_error("Invalid function call");
            }
            
            call->args = parse_argument_list();
            expect(ast::TokenType::RPAREN, "Expected ')'");
            expr = std::move(call);
        } else if (current_token_.type == ast::TokenType::DOT) {
            advance(); // consume '.'
            auto member = current_token_.value;
            expect(ast::TokenType::IDENTIFIER, "Expected member name");
            
            auto member_access = std::make_unique<ast::MemberAccess>();
            member_access->object = std::move(expr);
            member_access->member = member;
            expr = std::move(member_access);
        } else {
            break;
        }
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_primary() {
    if (current_token_.type == ast::TokenType::NUMBER) {
        double value = std::stod(current_token_.value);
        advance();
        auto number = std::make_unique<ast::NumberLiteral>();
        number->value = value;
        return number;
    }
    
    if (current_token_.type == ast::TokenType::STRING) {
        auto str_value = current_token_.value;
        advance();
        auto string_lit = std::make_unique<ast::StringLiteral>();
        string_lit->value = str_value;
        return string_lit;
    }
    
    if (current_token_.type == ast::TokenType::IDENTIFIER) {
        auto name = current_token_.value;
        advance();
        auto identifier = std::make_unique<ast::Identifier>();
        identifier->name = name;
        return identifier;
    }
    
    if (current_token_.type == ast::TokenType::LPAREN) {
        advance(); // consume '('
        auto expr = parse_expression();
        expect(ast::TokenType::RPAREN, "Expected ')'");
        return expr;
    }
    
    throw std::runtime_error("Unexpected token in expression: " + current_token_.value);
}

std::vector<std::unique_ptr<ast::Expression>> Parser::parse_argument_list() {
    std::vector<std::unique_ptr<ast::Expression>> arguments;
    
    if (current_token_.type != ast::TokenType::RPAREN) {
        do {
            arguments.push_back(parse_expression());
        } while (current_token_.type == ast::TokenType::COMMA && (advance(), true));
    }
    
    return arguments;
}

std::unordered_map<std::string, std::string> Parser::parse_parameter_list() {
    std::unordered_map<std::string, std::string> parameters;
    
    while (current_token_.type != ast::TokenType::RBRACE && current_token_.type != ast::TokenType::EOF_TOKEN) {
        auto param_name = current_token_.value;
        expect(ast::TokenType::IDENTIFIER, "Expected parameter name");
        expect(ast::TokenType::COLON, "Expected ':'");
        
        // For now, just store the next token as string value
        auto param_value = current_token_.value;
        advance(); // consume the value token
        parameters[param_name] = param_value;
        
        if (current_token_.type != ast::TokenType::COMMA) {
            break;
        }
        advance(); // consume ','
    }
    
    return parameters;
}

} // namespace dsl::parser
