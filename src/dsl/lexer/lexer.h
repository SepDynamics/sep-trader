#pragma once
#include <string>
#include <unordered_map>

#include "ast/nodes.h"

namespace dsl::lexer {

struct Token {
    ast::TokenType type;
    std::string value;
    size_t line;
    size_t column;
    
    Token(ast::TokenType t, const std::string& v, size_t l, size_t c)
        : type(t), value(v), line(l), column(c) {}
};

class Lexer {
private:
    std::string source;
    size_t position;
    size_t line;
    size_t column;
    char current_char;
    
    static std::unordered_map<std::string, ast::TokenType> keywords;
    
    void advance();
    void skip_whitespace();
    void skip_comment();
    Token read_number();
    Token read_string();
    Token read_identifier();
    char peek(size_t offset = 1);

public:
    Lexer(const std::string& source);
    Token next_token();
    Token peek_token();
    bool is_at_end() const;
};

} // namespace dsl::lexer
