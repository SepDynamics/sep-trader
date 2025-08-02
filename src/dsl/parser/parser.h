#pragma once

#include "dsl/ast/nodes.h"
#include <string>
#include <memory>

namespace sep::dsl::parser {

    class SEPParser {
    public:
        SEPParser();
        ~SEPParser();

        // Main parsing entry point
        std::unique_ptr<ast::Program> parse(const std::string& source);
        
        // Individual parsing methods for exploration
        std::unique_ptr<ast::PatternNode> parsePattern(const std::string& source);
        std::unique_ptr<ast::Expression> parseExpression(const std::string& source);
        
        // Error handling
        bool hasErrors() const;
        std::vector<std::string> getErrors() const;

    private:
        std::vector<std::string> errors_;
        
        // Internal parsing state (will expand during implementation)
        size_t current_pos_;
        std::string source_;
        
        // Tokenization helpers
        void skipWhitespace();
        std::string parseIdentifier();
        std::string parseString();
        double parseNumber();
        
        // Parsing helpers
        std::unique_ptr<ast::Expression> parseArithmeticExpression();
        std::unique_ptr<ast::Expression> parseFunctionCall();
        std::vector<std::pair<std::string, std::unique_ptr<ast::Expression>>> parseParameters();
    };

} // namespace sep::dsl::parser
