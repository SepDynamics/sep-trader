#include "parser.h"
#include <stdexcept>
#include <iostream>

namespace dsl::parser {

Parser::Parser(const std::string& source, bool optimize) 
    : lexer_(source), current_token_(ast::TokenType::EOF_TOKEN, "", 0, 0), enable_optimization_(optimize) {
    advance();
}

void Parser::advance() {
    current_token_ = lexer_.next_token();
}

void Parser::expect(ast::TokenType type, const std::string& message) {
    if (current_token_.type == type) {
        advance();
    } else {
        throw std::runtime_error(message + " at " + current_location().to_string());
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
        } else if (current_token_.type == ast::TokenType::FUNCTION) {
            // For now, we'll ignore top-level function declarations
            // In a proper implementation, we'd add them to the program structure
            parse_function_declaration();
        } else if (current_token_.type == ast::TokenType::ASYNC) {
            // For now, we'll ignore top-level async function declarations
            // In a proper implementation, we'd add them to the program structure
            parse_async_function_declaration();
        } else if (current_token_.type == ast::TokenType::IDENTIFIER) {
            // Handle top-level assignments or expressions
            // For now, we'll just parse and ignore them
            parse_statement();
        } else if (current_token_.type == ast::TokenType::FOR) {
            // Handle top-level for loops
            parse_statement();
        } else if (current_token_.type == ast::TokenType::WHILE) {
            // Handle top-level while loops
            parse_statement();
        } else {
            throw std::runtime_error("Unexpected token: " + current_token_.value + " at line " + std::to_string(current_token_.line));
        }
    }
    
    // Apply AST optimizations if enabled
    if (enable_optimization_) {
        optimizer::ASTOptimizer optimizer;
        optimizer.optimize(*program);
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
    
    // Check for inheritance
    if (current_token_.type == ast::TokenType::INHERITS) {
        advance(); // consume 'inherits'
        pattern_decl->parent_pattern = current_token_.value;
        expect(ast::TokenType::IDENTIFIER, "Expected parent pattern name after 'inherits'");
    }
    
    expect(ast::TokenType::LBRACE, "Expected '{'");
    
    // Parse input declarations
    if (current_token_.type == ast::TokenType::INPUT) {
        advance(); // consume 'input'
        expect(ast::TokenType::COLON, "Expected ':'");
        
        do {
            auto input_name = current_token_.value;
            expect(ast::TokenType::IDENTIFIER, "Expected input name");
            pattern_decl->inputs.push_back(input_name);
        } while (current_token_.type == ast::TokenType::COMMA && (advance(), true));
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
            expect(ast::TokenType::SEMICOLON, "Expected ';' after trigger expression");
        } else if (current_token_.value == "confidence") {
            advance(); // consume 'confidence'
            expect(ast::TokenType::COLON, "Expected ':'");
            signal_decl->confidence = parse_expression();
            expect(ast::TokenType::SEMICOLON, "Expected ';' after confidence expression");
        } else if (current_token_.value == "action") {
            advance(); // consume 'action'
            expect(ast::TokenType::COLON, "Expected ':'");
            signal_decl->action = current_token_.value;
            expect(ast::TokenType::STRING, "Expected action name as a string literal");
            expect(ast::TokenType::SEMICOLON, "Expected ';' after action value");
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
        // Consume optional semicolon after statement
        if (current_token_.type == ast::TokenType::SEMICOLON) {
            advance();
        }
    }
    
    return statements;
}

std::unique_ptr<ast::Statement> Parser::parse_statement() {
    // Check for evolve statement
    if (current_token_.type == ast::TokenType::EVOLVE) {
        return parse_evolve_statement();
    }

    // Check for if statement
    if (current_token_.type == ast::TokenType::IF) {
        return parse_if_statement();
    }

    // Check for while statement
    if (current_token_.type == ast::TokenType::WHILE) {
        return parse_while_statement();
    }

    // Check for for statement
    if (current_token_.type == ast::TokenType::FOR) {
        return parse_for_statement();
    }

    // Check for function declaration
    if (current_token_.type == ast::TokenType::FUNCTION) {
        return parse_function_declaration();
    }

    // Check for return statement
    if (current_token_.type == ast::TokenType::RETURN) {
        return parse_return_statement();
    }

    // Check for import statement
    if (current_token_.type == ast::TokenType::IMPORT) {
        return parse_import_statement();
    }

    // Check for export statement
    if (current_token_.type == ast::TokenType::EXPORT) {
        return parse_export_statement();
    }

    // Check for async function declaration
    if (current_token_.type == ast::TokenType::ASYNC) {
        return parse_async_function_declaration();
    }

    // Check for try statement
    if (current_token_.type == ast::TokenType::TRY) {
        return parse_try_statement();
    }

    // Check for throw statement
    if (current_token_.type == ast::TokenType::THROW) {
        return parse_throw_statement();
    }

    // Check for assignment (identifier [: type] = expression)
    if (current_token_.type == ast::TokenType::IDENTIFIER) {
        // Look ahead to see if this is an assignment (with or without type annotation)
        lexer::Token next = lexer_.peek_token();
        if (next.type == ast::TokenType::ASSIGN || next.type == ast::TokenType::COLON) {
            auto name = current_token_.value;
            advance(); // consume identifier
            
            // Check for type annotation
            ast::TypeAnnotation var_type = ast::TypeAnnotation::INFERRED;
            if (current_token_.type == ast::TokenType::COLON) {
                advance(); // consume ':'
                var_type = parse_type_annotation();
            }
            
            expect(ast::TokenType::ASSIGN, "Expected '='");
            auto value = parse_expression();
            
            auto assignment = std::make_unique<ast::Assignment>();
            assignment->name = name;
            assignment->type = var_type;
            assignment->value = std::move(value);
            assignment->location = ast::SourceLocation(next.line, next.column); // Use the identifier location
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
    return parse_precedence(Precedence::LOWEST);
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
    
    while (current_token_.type == ast::TokenType::MULTIPLY || current_token_.type == ast::TokenType::DIVIDE || current_token_.type == ast::TokenType::MODULO) {
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
        std::string op = current_token_.value;
        advance();
        auto right = parse_unary();
        
        auto unary = std::make_unique<ast::UnaryOp>();
        unary->op = op;
        unary->right = std::move(right);
        return unary;
    }
    
    return parse_call();
}

std::unique_ptr<ast::Expression> Parser::parse_call() {
    auto expr = parse_primary();
    
    // This loop now only handles member access, e.g., pattern.result
    while (current_token_.type == ast::TokenType::DOT) {
        advance(); // consume '.'
        auto member = current_token_.value;
        expect(ast::TokenType::IDENTIFIER, "Expected member name");
        
        auto member_access = std::make_unique<ast::MemberAccess>();
        member_access->object = std::move(expr);
        member_access->member = member;
        expr = std::move(member_access);
    }
    
    return expr;
}

std::unique_ptr<ast::Expression> Parser::parse_primary() {
    if (current_token_.type == ast::TokenType::NUMBER) {
        auto location = current_location();
        double value = std::stod(current_token_.value);
        advance();
        auto number = std::make_unique<ast::NumberLiteral>();
        number->value = value;
        number->location = location;
        return number;
    }
    
    if (current_token_.type == ast::TokenType::STRING) {
        auto str_value = current_token_.value;
        advance();
        auto string_lit = std::make_unique<ast::StringLiteral>();
        string_lit->value = str_value;
        return string_lit;
    }
    
    if (current_token_.type == ast::TokenType::BOOLEAN) {
        bool value = (current_token_.value == "true");
        advance();
        auto boolean_lit = std::make_unique<ast::BooleanLiteral>();
        boolean_lit->value = value;
        return boolean_lit;
    }
    
    if (current_token_.type == ast::TokenType::IDENTIFIER) {
        std::string name = current_token_.value;
        advance(); // Consume the identifier
        
        // If the next token is '(', it's a function call.
        if (current_token_.type == ast::TokenType::LPAREN) {
            advance(); // Consume '('
            auto call = std::make_unique<ast::Call>();
            call->callee = name;
            call->args = parse_argument_list();
            expect(ast::TokenType::RPAREN, "Expected ')' after function arguments.");
            return call;
        }
        
        // Otherwise, it's just a variable identifier.
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
    
    if (current_token_.type == ast::TokenType::WEIGHTED_SUM) {
       return parse_weighted_sum();
    }
    
    if (current_token_.type == ast::TokenType::AWAIT) {
        return parse_await_expression();
    }
    
    std::cout << "[DEBUG] Current token: " << static_cast<int>(current_token_.type) << " value: " << current_token_.value << std::endl;
    std::cout << "[DEBUG] IF token type: " << static_cast<int>(ast::TokenType::IF) << std::endl;
    
    if (current_token_.type == ast::TokenType::IF) {
        std::cout << "[DEBUG] Parsing if expression" << std::endl;
        return parse_if_expression();
    }
    
    // Array literals
    if (current_token_.type == ast::TokenType::LBRACKET) {
        auto location = current_location();
        advance(); // consume '['
        
        auto array = std::make_unique<ast::ArrayLiteral>();
        array->location = location;
        
        // Parse array elements
        if (current_token_.type != ast::TokenType::RBRACKET) {
            do {
                array->elements.push_back(parse_expression());
            } while (current_token_.type == ast::TokenType::COMMA && (advance(), true));
        }
        
        expect(ast::TokenType::RBRACKET, "Expected ']' after array elements");
        return array;
    }
    
    // Vector literals: vec2(x, y), vec3(x, y, z), vec4(x, y, z, w)
    if (current_token_.type == ast::TokenType::IDENTIFIER && 
        (current_token_.value == "vec2" || current_token_.value == "vec3" || current_token_.value == "vec4")) {
        
        std::string vec_type = current_token_.value;
        advance(); // consume vector type
        expect(ast::TokenType::LPAREN, "Expected '(' after vector type");
        
        auto vector_lit = std::make_unique<ast::VectorLiteral>();
        
        // Parse vector components
        if (current_token_.type != ast::TokenType::RPAREN) {
            do {
                vector_lit->components.push_back(parse_expression());
            } while (current_token_.type == ast::TokenType::COMMA && (advance(), true));
        }
        
        expect(ast::TokenType::RPAREN, "Expected ')' after vector components");
        
        // Validate component count
        size_t expected_components = (vec_type == "vec2") ? 2 : (vec_type == "vec3") ? 3 : 4;
        if (vector_lit->components.size() != expected_components) {
            throw std::runtime_error(vec_type + " requires exactly " + std::to_string(expected_components) + " components");
        }
        
        return vector_lit;
    }
    
    throw std::runtime_error("Unexpected token in expression: " + current_token_.value);
}

std::unique_ptr<ast::WeightedSum> Parser::parse_weighted_sum() {
   expect(ast::TokenType::WEIGHTED_SUM, "Expected 'weighted_sum' keyword.");
   expect(ast::TokenType::LBRACE, "Expected '{' after 'weighted_sum'.");

   auto weighted_sum = std::make_unique<ast::WeightedSum>();

   // Parse expressions (comma-separated, semicolon-separated, or newline-separated)
   while (current_token_.type != ast::TokenType::RBRACE && current_token_.type != ast::TokenType::EOF_TOKEN) {
   auto expr = parse_expression();
   weighted_sum->expressions.push_back(std::move(expr));
   
   // Accept optional comma, semicolon, or continue to next expression
   if (current_token_.type == ast::TokenType::COMMA || current_token_.type == ast::TokenType::SEMICOLON) {
   advance(); // consume separator
   }
   // Otherwise, continue parsing the next expression if not at closing brace
   }

   expect(ast::TokenType::RBRACE, "Expected '}' to close 'weighted_sum' block.");
   return weighted_sum;
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

std::unique_ptr<ast::EvolveStatement> Parser::parse_evolve_statement() {
    expect(ast::TokenType::EVOLVE, "Expected 'evolve' keyword.");
    expect(ast::TokenType::WHEN, "Expected 'when' after 'evolve'.");
    
    auto evolve_stmt = std::make_unique<ast::EvolveStatement>();
    evolve_stmt->condition = parse_expression();
    
    expect(ast::TokenType::LBRACE, "Expected '{' after evolve condition.");
    evolve_stmt->body = parse_block();
    expect(ast::TokenType::RBRACE, "Expected '}' to close evolve block.");
    
    return evolve_stmt;
}

std::unique_ptr<ast::IfStatement> Parser::parse_if_statement() {
    expect(ast::TokenType::IF, "Expected 'if' keyword.");
    expect(ast::TokenType::LPAREN, "Expected '(' after 'if'.");
    
    auto if_stmt = std::make_unique<ast::IfStatement>();
    if_stmt->condition = parse_expression();
    
    expect(ast::TokenType::RPAREN, "Expected ')' after condition.");
    expect(ast::TokenType::LBRACE, "Expected '{' after if condition.");
    if_stmt->then_branch = parse_block();
    expect(ast::TokenType::RBRACE, "Expected '}' to close if block.");
    
    // Optional else branch
    if (current_token_.type == ast::TokenType::ELSE) {
        advance(); // consume 'else'
        expect(ast::TokenType::LBRACE, "Expected '{' after 'else'.");
        if_stmt->else_branch = parse_block();
        expect(ast::TokenType::RBRACE, "Expected '}' to close else block.");
    }
    
    return if_stmt;
}

std::unique_ptr<ast::WhileStatement> Parser::parse_while_statement() {
    expect(ast::TokenType::WHILE, "Expected 'while' keyword.");
    expect(ast::TokenType::LPAREN, "Expected '(' after 'while'.");
    
    auto while_stmt = std::make_unique<ast::WhileStatement>();
    while_stmt->condition = parse_expression();
    
    expect(ast::TokenType::RPAREN, "Expected ')' after condition.");
    expect(ast::TokenType::LBRACE, "Expected '{' after while condition.");
    while_stmt->body = parse_block();
    expect(ast::TokenType::RBRACE, "Expected '}' to close while block.");
    
    return while_stmt;
}

std::unique_ptr<ast::ForStatement> Parser::parse_for_statement() {
    expect(ast::TokenType::FOR, "Expected 'for' keyword.");
    expect(ast::TokenType::LPAREN, "Expected '(' after 'for'.");
    
    auto for_stmt = std::make_unique<ast::ForStatement>();
    
    // Parse loop variable
    if (current_token_.type != ast::TokenType::IDENTIFIER) {
        throw std::runtime_error("Expected variable name in for loop.");
    }
    for_stmt->variable = current_token_.value;
    advance(); // consume variable name
    
    expect(ast::TokenType::IN, "Expected 'in' after for loop variable.");
    
    // Parse iterable expression
    for_stmt->iterable = parse_expression();
    
    expect(ast::TokenType::RPAREN, "Expected ')' after for loop.");
    expect(ast::TokenType::LBRACE, "Expected '{' after for loop.");
    for_stmt->body = parse_block();
    expect(ast::TokenType::RBRACE, "Expected '}' to close for loop block.");
    
    return for_stmt;
}

// Type annotation helper functions
ast::TypeAnnotation Parser::parse_type_annotation() {
    switch (current_token_.type) {
        case ast::TokenType::NUMBER_TYPE:
            advance();
            return ast::TypeAnnotation::NUMBER;
        case ast::TokenType::STRING_TYPE:
            advance();
            return ast::TypeAnnotation::STRING;
        case ast::TokenType::BOOL_TYPE:
            advance();
            return ast::TypeAnnotation::BOOL;
        case ast::TokenType::PATTERN_TYPE:
            advance();
            return ast::TypeAnnotation::PATTERN;
        case ast::TokenType::VOID_TYPE:
            advance();
            return ast::TypeAnnotation::VOID;
        case ast::TokenType::VEC2_TYPE:
            advance();
            return ast::TypeAnnotation::VEC2;
        case ast::TokenType::VEC3_TYPE:
            advance();
            return ast::TypeAnnotation::VEC3;
        case ast::TokenType::VEC4_TYPE:
            advance();
            return ast::TypeAnnotation::VEC4;
        default:
            return ast::TypeAnnotation::INFERRED;
    }
}

bool Parser::is_type_token(ast::TokenType type) {
    return type == ast::TokenType::NUMBER_TYPE ||
           type == ast::TokenType::STRING_TYPE ||
           type == ast::TokenType::BOOL_TYPE ||
           type == ast::TokenType::PATTERN_TYPE ||
           type == ast::TokenType::VOID_TYPE;
}

// Source location helper functions
ast::SourceLocation Parser::current_location() const {
    return ast::SourceLocation(current_token_.line, current_token_.column);
}

void Parser::set_location(ast::Node& node) const {
    node.location = current_location();
}

// Operator precedence implementation
const std::unordered_map<ast::TokenType, Parser::OperatorInfo>& Parser::get_operator_table() const {
    static const std::unordered_map<ast::TokenType, OperatorInfo> operator_table = {
        // Logical operators
        {ast::TokenType::OR, {Precedence::LOGICAL_OR, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::AND, {Precedence::LOGICAL_AND, OperatorInfo::Associativity::LEFT, false, false}},
        
        // Equality operators
        {ast::TokenType::EQ, {Precedence::EQUALITY, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::NE, {Precedence::EQUALITY, OperatorInfo::Associativity::LEFT, false, false}},
        
        // Relational operators
        {ast::TokenType::LT, {Precedence::RELATIONAL, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::LE, {Precedence::RELATIONAL, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::GT, {Precedence::RELATIONAL, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::GE, {Precedence::RELATIONAL, OperatorInfo::Associativity::LEFT, false, false}},
        
        // Additive operators
        {ast::TokenType::PLUS, {Precedence::ADDITIVE, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::MINUS, {Precedence::ADDITIVE, OperatorInfo::Associativity::LEFT, false, false}},
        
        // Multiplicative operators
        {ast::TokenType::MULTIPLY, {Precedence::MULTIPLICATIVE, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::DIVIDE, {Precedence::MULTIPLICATIVE, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::MODULO, {Precedence::MULTIPLICATIVE, OperatorInfo::Associativity::LEFT, false, false}},
        
        // Unary operators
        {ast::TokenType::NOT, {Precedence::UNARY, OperatorInfo::Associativity::RIGHT, true, false}},
        
        // Function calls and member access
        {ast::TokenType::LPAREN, {Precedence::CALL, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::DOT, {Precedence::MEMBER, OperatorInfo::Associativity::LEFT, false, false}},
        {ast::TokenType::LBRACKET, {Precedence::POSTFIX, OperatorInfo::Associativity::LEFT, false, true}},
    };
    return operator_table;
}

Parser::Precedence Parser::get_precedence(ast::TokenType token_type) const {
    const auto& table = get_operator_table();
    auto it = table.find(token_type);
    return (it != table.end()) ? it->second.precedence : Precedence::LOWEST;
}

Parser::OperatorInfo::Associativity Parser::get_associativity(ast::TokenType token_type) const {
    const auto& table = get_operator_table();
    auto it = table.find(token_type);
    return (it != table.end()) ? it->second.associativity : OperatorInfo::Associativity::LEFT;
}

std::unique_ptr<ast::Expression> Parser::parse_precedence(Precedence precedence) {
    // Parse left side (prefix)
    std::unique_ptr<ast::Expression> left;
    
    switch (current_token_.type) {
        case ast::TokenType::NUMBER: {
            auto location = current_location();
            double value = std::stod(current_token_.value);
            advance();
            auto number = std::make_unique<ast::NumberLiteral>();
            number->value = value;
            number->location = location;
            left = std::move(number);
            break;
        }
        case ast::TokenType::STRING: {
            auto str_value = current_token_.value;
            advance();
            auto string = std::make_unique<ast::StringLiteral>();
            string->value = str_value;
            left = std::move(string);
            break;
        }
        case ast::TokenType::BOOLEAN: {
            bool value = current_token_.value == "true";
            advance();
            auto boolean_lit = std::make_unique<ast::BooleanLiteral>();
            boolean_lit->value = value;
            left = std::move(boolean_lit);
            break;
        }
        case ast::TokenType::IDENTIFIER: {
            std::string name = current_token_.value;
            advance();
            
            // Check for function call
            if (current_token_.type == ast::TokenType::LPAREN) {
                advance(); // consume '('
                auto call = std::make_unique<ast::Call>();
                call->callee = name;
                call->args = parse_argument_list();
                expect(ast::TokenType::RPAREN, "Expected ')' after function arguments.");
                left = std::move(call);
            } else {
                auto identifier = std::make_unique<ast::Identifier>();
                identifier->name = name;
                left = std::move(identifier);
            }
            break;
        }
        case ast::TokenType::LPAREN: {
            advance(); // consume '('
            left = parse_precedence(Precedence::LOWEST);
            expect(ast::TokenType::RPAREN, "Expected ')'");
            break;
        }
        case ast::TokenType::NOT:
        case ast::TokenType::MINUS: {
            std::string op = current_token_.value;
            advance();
            auto right = parse_precedence(Precedence::UNARY);
            auto unary = std::make_unique<ast::UnaryOp>();
            unary->op = op;
            unary->right = std::move(right);
            left = std::move(unary);
            break;
        }
        case ast::TokenType::LBRACKET: {
            auto location = current_location();
            advance(); // consume '['
            
            auto array = std::make_unique<ast::ArrayLiteral>();
            array->location = location;
            
            // Parse array elements
            if (current_token_.type != ast::TokenType::RBRACKET) {
                do {
                    array->elements.push_back(parse_expression());
                } while (current_token_.type == ast::TokenType::COMMA && (advance(), true));
            }
            
            expect(ast::TokenType::RBRACKET, "Expected ']' after array elements");
            left = std::move(array);
            break;
        }
        case ast::TokenType::IF: {
            left = parse_if_expression();
            break;
        }
        case ast::TokenType::WEIGHTED_SUM: {
            left = parse_weighted_sum();
            break;
        }
        case ast::TokenType::AWAIT: {
            left = parse_await_expression();
            break;
        }
        default:
            throw std::runtime_error("Unexpected token in expression: " + current_token_.value);
    }
    
    // Parse right side (infix and postfix)
    while (precedence < get_precedence(current_token_.type) || current_token_.type == ast::TokenType::LBRACKET) {
        ast::TokenType op_type = current_token_.type;
        
        // Handle array access [index]
        if (current_token_.type == ast::TokenType::LBRACKET) {
            advance(); // consume '['
            auto array_access = std::make_unique<ast::ArrayAccess>();
            array_access->array = std::move(left);
            array_access->index = parse_expression();
            expect(ast::TokenType::RBRACKET, "Expected ']' after array index");
            left = std::move(array_access);
            continue;
        }
        
        std::string op = current_token_.value;
        advance();
        
        auto binary = std::make_unique<ast::BinaryOp>();
        binary->left = std::move(left);
        binary->op = op;
        
        // Handle associativity correctly
        Precedence next_precedence;
        if (get_associativity(op_type) == OperatorInfo::Associativity::RIGHT) {
            // Right-associative: use same precedence for recursive call
            next_precedence = get_precedence(op_type);
        } else {
            // Left-associative: use higher precedence for recursive call
            next_precedence = static_cast<Precedence>(static_cast<int>(get_precedence(op_type)) + 1);
        }
        
        binary->right = parse_precedence(next_precedence);
        left = std::move(binary);
    }
    
    return left;
}

std::unique_ptr<ast::FunctionDeclaration> Parser::parse_function_declaration() {
    expect(ast::TokenType::FUNCTION, "Expected 'function' keyword.");
    
    auto name = current_token_.value;
    expect(ast::TokenType::IDENTIFIER, "Expected function name.");
    
    expect(ast::TokenType::LPAREN, "Expected '(' after function name.");
    
    auto func_decl = std::make_unique<ast::FunctionDeclaration>();
    func_decl->name = name;
    
    // Parse parameter list
    if (current_token_.type != ast::TokenType::RPAREN) {
        do {
            auto param_name = current_token_.value;
            expect(ast::TokenType::IDENTIFIER, "Expected parameter name");
            
            // Check for type annotation
            ast::TypeAnnotation param_type = ast::TypeAnnotation::INFERRED;
            if (current_token_.type == ast::TokenType::COLON) {
                advance(); // consume ':'
                param_type = parse_type_annotation();
            }
            
            func_decl->parameters.push_back(std::make_pair(param_name, param_type));
        } while (current_token_.type == ast::TokenType::COMMA && (advance(), true));
    }
    
    expect(ast::TokenType::RPAREN, "Expected ')' after parameters.");
    
    // Check for return type annotation
    if (current_token_.type == ast::TokenType::COLON) {
        advance(); // consume ':'
        func_decl->return_type = parse_type_annotation();
    }
    
    expect(ast::TokenType::LBRACE, "Expected '{' before function body.");
    
    // Parse function body
    func_decl->body = parse_block();
    expect(ast::TokenType::RBRACE, "Expected '}' after function body.");
    
    return func_decl;
}

std::unique_ptr<ast::ReturnStatement> Parser::parse_return_statement() {
    expect(ast::TokenType::RETURN, "Expected 'return' keyword.");
    
    auto return_stmt = std::make_unique<ast::ReturnStatement>();
    
    // Parse return value if present (not a void return)
    if (current_token_.type != ast::TokenType::SEMICOLON) {
        return_stmt->value = parse_expression();
    }
    
    return return_stmt;
}

std::unique_ptr<ast::ImportStatement> Parser::parse_import_statement() {
    expect(ast::TokenType::IMPORT, "Expected 'import' keyword.");
    
    auto import_stmt = std::make_unique<ast::ImportStatement>();
    
    // Parse module path (string literal)
    if (current_token_.type == ast::TokenType::STRING) {
        import_stmt->module_path = current_token_.value;
        advance();
    } else {
        throw std::runtime_error("Expected string literal for module path");
    }
    
    // Check for specific imports (import "module" { pattern1, pattern2 })
    if (current_token_.type == ast::TokenType::LBRACE) {
        advance(); // consume '{'
        
        while (current_token_.type != ast::TokenType::RBRACE && current_token_.type != ast::TokenType::EOF_TOKEN) {
            if (current_token_.type == ast::TokenType::IDENTIFIER) {
                import_stmt->imports.push_back(current_token_.value);
                advance();
                
                if (current_token_.type == ast::TokenType::COMMA) {
                    advance(); // consume ','
                }
            } else {
                throw std::runtime_error("Expected identifier in import list");
            }
        }
        
        expect(ast::TokenType::RBRACE, "Expected '}' after import list");
    }
    
    return import_stmt;
}

std::unique_ptr<ast::ExportStatement> Parser::parse_export_statement() {
    expect(ast::TokenType::EXPORT, "Expected 'export' keyword.");
    
    auto export_stmt = std::make_unique<ast::ExportStatement>();
    
    expect(ast::TokenType::LBRACE, "Expected '{' after 'export'");
    
    while (current_token_.type != ast::TokenType::RBRACE && current_token_.type != ast::TokenType::EOF_TOKEN) {
        if (current_token_.type == ast::TokenType::IDENTIFIER) {
            export_stmt->exports.push_back(current_token_.value);
            advance();
            
            if (current_token_.type == ast::TokenType::COMMA) {
                advance(); // consume ','
            }
        } else {
            throw std::runtime_error("Expected identifier in export list");
        }
    }
    
    expect(ast::TokenType::RBRACE, "Expected '}' after export list");
    
    return export_stmt;
}

std::unique_ptr<ast::AsyncFunctionDeclaration> Parser::parse_async_function_declaration() {
    expect(ast::TokenType::ASYNC, "Expected 'async' keyword.");
    expect(ast::TokenType::FUNCTION, "Expected 'function' after 'async'.");
    
    auto func_name = current_token_.value;
    expect(ast::TokenType::IDENTIFIER, "Expected function name.");
    
    expect(ast::TokenType::LPAREN, "Expected '(' after function name.");
    
    auto async_func = std::make_unique<ast::AsyncFunctionDeclaration>();
    async_func->name = func_name;
    
    // Parse parameter list
    if (current_token_.type != ast::TokenType::RPAREN) {
        do {
            auto param_name = current_token_.value;
            expect(ast::TokenType::IDENTIFIER, "Expected parameter name");
            
            // Check for type annotation
            ast::TypeAnnotation param_type = ast::TypeAnnotation::INFERRED;
            if (current_token_.type == ast::TokenType::COLON) {
                advance(); // consume ':'
                param_type = parse_type_annotation();
            }
            
            async_func->parameters.push_back(std::make_pair(param_name, param_type));
        } while (current_token_.type == ast::TokenType::COMMA && (advance(), true));
    }
    
    expect(ast::TokenType::RPAREN, "Expected ')' after parameter list.");
    
    // Check for return type annotation
    if (current_token_.type == ast::TokenType::COLON) {
        advance(); // consume ':'
        async_func->return_type = parse_type_annotation();
    }
    
    expect(ast::TokenType::LBRACE, "Expected '{' to start function body.");
    
    async_func->body = parse_block();
    
    expect(ast::TokenType::RBRACE, "Expected '}' to end function body.");
    
    return async_func;
}

std::unique_ptr<ast::AwaitExpression> Parser::parse_await_expression() {
    expect(ast::TokenType::AWAIT, "Expected 'await' keyword.");
    
    auto await_expr = std::make_unique<ast::AwaitExpression>();
    await_expr->expression = parse_unary(); // Parse the expression being awaited
    
    return await_expr;
}

std::unique_ptr<ast::TryStatement> Parser::parse_try_statement() {
    expect(ast::TokenType::TRY, "Expected 'try' keyword.");
    expect(ast::TokenType::LBRACE, "Expected '{' after 'try'.");
    
    auto try_stmt = std::make_unique<ast::TryStatement>();
    try_stmt->try_body = parse_block();
    
    expect(ast::TokenType::RBRACE, "Expected '}' after try block.");
    
    // Parse catch block (required)
    expect(ast::TokenType::CATCH, "Expected 'catch' after try block.");
    expect(ast::TokenType::LPAREN, "Expected '(' after 'catch'.");
    
    if (current_token_.type == ast::TokenType::IDENTIFIER) {
        try_stmt->catch_variable = current_token_.value;
        advance();
    } else {
        throw std::runtime_error("Expected catch variable name");
    }
    
    expect(ast::TokenType::RPAREN, "Expected ')' after catch variable.");
    expect(ast::TokenType::LBRACE, "Expected '{' to start catch block.");
    
    try_stmt->catch_body = parse_block();
    
    expect(ast::TokenType::RBRACE, "Expected '}' after catch block.");
    
    // Parse optional finally block
    if (current_token_.type == ast::TokenType::FINALLY) {
        advance(); // consume 'finally'
        expect(ast::TokenType::LBRACE, "Expected '{' after 'finally'.");
        try_stmt->finally_body = parse_block();
        expect(ast::TokenType::RBRACE, "Expected '}' after finally block.");
    }
    
    return try_stmt;
}

std::unique_ptr<ast::ThrowStatement> Parser::parse_throw_statement() {
    expect(ast::TokenType::THROW, "Expected 'throw' keyword.");
    
    auto throw_stmt = std::make_unique<ast::ThrowStatement>();
    throw_stmt->expression = parse_expression();
    
    return throw_stmt;
}

std::unique_ptr<ast::IfExpression> Parser::parse_if_expression() {
    expect(ast::TokenType::IF, "Expected 'if' keyword.");
    expect(ast::TokenType::LPAREN, "Expected '(' after 'if'.");
    
    auto if_expr = std::make_unique<ast::IfExpression>();
    if_expr->condition = parse_expression();
    
    expect(ast::TokenType::RPAREN, "Expected ')' after condition.");
    expect(ast::TokenType::LBRACE, "Expected '{' after if condition.");
    
    // Parse the then expression (should be a single expression)
    if_expr->then_expr = parse_expression();
    
    expect(ast::TokenType::RBRACE, "Expected '}' to close if block.");
    expect(ast::TokenType::ELSE, "Expected 'else' after if expression.");
    
    // Check if this is an 'else if' chain
    if (current_token_.type == ast::TokenType::IF) {
        // This is an 'else if', recursively parse the next if expression
        if_expr->else_expr = parse_if_expression();
    } else {
        // This is a regular else clause
        expect(ast::TokenType::LBRACE, "Expected '{' after 'else'.");
        if_expr->else_expr = parse_expression();
        expect(ast::TokenType::RBRACE, "Expected '}' to close else block.");
    }
    
    return if_expr;
}

} // namespace dsl::parser
