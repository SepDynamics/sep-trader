#include "lexer.h"
#include <cctype>
#include <stdexcept>

namespace dsl::lexer {

    std::unordered_map<std::string, ast::TokenType> Lexer::keywords = {
        {"pattern", ast::TokenType::PATTERN},
        {"stream", ast::TokenType::STREAM},
        {"signal", ast::TokenType::SIGNAL},
        {"memory", ast::TokenType::MEMORY},
        {"inherits", ast::TokenType::INHERITS},
        {"from", ast::TokenType::FROM},
        {"when", ast::TokenType::WHEN},
        {"using", ast::TokenType::USING},
        {"input", ast::TokenType::INPUT},
        {"output", ast::TokenType::OUTPUT},
        {"weighted_sum", ast::TokenType::WEIGHTED_SUM},
        {"evolve", ast::TokenType::EVOLVE},
        // Keep these as function identifiers, NOT keywords
        // Control flow keywords
        {"if", ast::TokenType::IF},
        {"else", ast::TokenType::ELSE},
        {"while", ast::TokenType::WHILE},
        {"function", ast::TokenType::FUNCTION},
        {"return", ast::TokenType::RETURN}};

    Lexer::Lexer(const std::string& source) : source(source), position(0), line(1), column(1)
    {
        current_char = position < source.length() ? source[position] : '\0';
    }

void Lexer::advance() {
    if (current_char == '\n') {
        line++;
        column = 1;
    } else {
        column++;
    }
    
    position++;
    current_char = position < source.length() ? source[position] : '\0';
}

void Lexer::skip_whitespace() {
    while (std::isspace(current_char)) {
        advance();
    }
}

void Lexer::skip_comment() {
    if (current_char == '/' && peek() == '/') {
        // Skip single-line comment
        while (current_char != '\n' && current_char != '\0') {
            advance();
        }
    } else if (current_char == '/' && peek() == '*') {
        // Skip multi-line comment
        advance(); // skip '/'
        advance(); // skip '*'
        
        while (!(current_char == '*' && peek() == '/') && current_char != '\0') {
            advance();
        }
        
        if (current_char != '\0') {
            advance(); // skip '*'
            advance(); // skip '/'
        }
    }
}

Token Lexer::read_number() {
    size_t start_line = line;
    size_t start_column = column;
    std::string number;
    
    while (std::isdigit(current_char)) {
        number += current_char;
        advance();
    }
    
    if (current_char == '.') {
        number += current_char;
        advance();
        
        while (std::isdigit(current_char)) {
            number += current_char;
            advance();
        }
    }
    
    return Token(ast::TokenType::NUMBER, number, start_line, start_column);
}

Token Lexer::read_string() {
    size_t start_line = line;
    size_t start_column = column;
    std::string str;
    char quote_char = current_char;
    
    advance(); // skip opening quote
    
    while (current_char != quote_char && current_char != '\0') {
        if (current_char == '\\') {
            advance();
            switch (current_char) {
                case 'n': str += '\n'; break;
                case 't': str += '\t'; break;
                case 'r': str += '\r'; break;
                case '\\': str += '\\'; break;
                case '"': str += '"'; break;
                case '\'': str += '\''; break;
                default: str += current_char; break;
            }
        } else {
            str += current_char;
        }
        advance();
    }
    
    if (current_char == quote_char) {
        advance(); // skip closing quote
    }
    
    return Token(ast::TokenType::STRING, str, start_line, start_column);
}

Token Lexer::read_identifier() {
    size_t start_line = line;
    size_t start_column = column;
    std::string identifier;
    
    while (std::isalnum(current_char) || current_char == '_') {
        identifier += current_char;
        advance();
    }
    
    // Check if it's a keyword
    auto keyword_it = keywords.find(identifier);
    ast::TokenType type = (keyword_it != keywords.end()) 
        ? keyword_it->second 
        : ast::TokenType::IDENTIFIER;
    
    return Token(type, identifier, start_line, start_column);
}

char Lexer::peek(size_t offset) {
    size_t peek_pos = position + offset;
    return peek_pos < source.length() ? source[peek_pos] : '\0';
}

Token Lexer::next_token() {
    while (current_char != '\0') {
        skip_whitespace();
        
        if (current_char == '\0') break;
        
        // Skip comments
        if (current_char == '/' && (peek() == '/' || peek() == '*')) {
            skip_comment();
            continue;
        }
        
        size_t token_line = line;
        size_t token_column = column;
        
        // Numbers
        if (std::isdigit(current_char)) {
            return read_number();
        }
        
        // Strings
        if (current_char == '"' || current_char == '\'') {
            return read_string();
        }
        
        // Identifiers and keywords
        if (std::isalpha(current_char) || current_char == '_') {
            return read_identifier();
        }
        
        // Two-character operators
        if (current_char == '=' && peek() == '=') {
            advance(); advance();
            return Token(ast::TokenType::EQ, "==", token_line, token_column);
        }
        if (current_char == '!' && peek() == '=') {
            advance(); advance();
            return Token(ast::TokenType::NE, "!=", token_line, token_column);
        }
        if (current_char == '<' && peek() == '=') {
            advance(); advance();
            return Token(ast::TokenType::LE, "<=", token_line, token_column);
        }
        if (current_char == '>' && peek() == '=') {
            advance(); advance();
            return Token(ast::TokenType::GE, ">=", token_line, token_column);
        }
        if (current_char == '&' && peek() == '&') {
            advance(); advance();
            return Token(ast::TokenType::AND, "&&", token_line, token_column);
        }
        if (current_char == '|' && peek() == '|') {
            advance(); advance();
            return Token(ast::TokenType::OR, "||", token_line, token_column);
        }
        
        // Single-character tokens
        switch (current_char) {
            case '{': advance(); return Token(ast::TokenType::LBRACE, "{", token_line, token_column);
            case '}': advance(); return Token(ast::TokenType::RBRACE, "}", token_line, token_column);
            case '(': advance(); return Token(ast::TokenType::LPAREN, "(", token_line, token_column);
            case ')': advance(); return Token(ast::TokenType::RPAREN, ")", token_line, token_column);
            case ';': advance(); return Token(ast::TokenType::SEMICOLON, ";", token_line, token_column);
            case ':': advance(); return Token(ast::TokenType::COLON, ":", token_line, token_column);
            case ',': advance(); return Token(ast::TokenType::COMMA, ",", token_line, token_column);
            case '.': advance(); return Token(ast::TokenType::DOT, ".", token_line, token_column);
            case '=': advance(); return Token(ast::TokenType::ASSIGN, "=", token_line, token_column);
            case '+': advance(); return Token(ast::TokenType::PLUS, "+", token_line, token_column);
            case '-': advance(); return Token(ast::TokenType::MINUS, "-", token_line, token_column);
            case '*': advance(); return Token(ast::TokenType::MULTIPLY, "*", token_line, token_column);
            case '/': advance(); return Token(ast::TokenType::DIVIDE, "/", token_line, token_column);
            case '<': advance(); return Token(ast::TokenType::LT, "<", token_line, token_column);
            case '>': advance(); return Token(ast::TokenType::GT, ">", token_line, token_column);
            case '!': advance(); return Token(ast::TokenType::NOT, "!", token_line, token_column);
            default:
                throw std::runtime_error("Unexpected character: " + std::string(1, current_char));
        }
    }
    
    return Token(ast::TokenType::EOF_TOKEN, "", line, column);
}

Token Lexer::peek_token() {
    size_t saved_pos = position;
    size_t saved_line = line;
    size_t saved_column = column;
    char saved_char = current_char;
    
    Token token = next_token();
    
    position = saved_pos;
    line = saved_line;
    column = saved_column;
    current_char = saved_char;
    
    return token;
}

bool Lexer::is_at_end() const {
    return current_char == '\0';
}

} // namespace dsl::lexer
