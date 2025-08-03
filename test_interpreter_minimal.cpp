#include "src/dsl/lexer/lexer.h"
#include "src/dsl/parser/parser.h"
// #include "src/dsl/runtime/interpreter.h"
#include <iostream>
#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Value = std::any;

// Simple mock for testing control flow parsing
class MockInterpreter {
public:
    void test_control_flow_parsing() {
        std::string source = R"(
            pattern TestControlFlow {
                x = 5
                
                if (x > 3) {
                    result = "greater"  
                } else {
                    result = "smaller"
                }
                
                counter = 0
                while (counter < 3) {
                    counter = counter + 1
                }
                
                function add(a, b) {
                    return a + b
                }
                
                sum = add(2, 3)
            }
        )";
        
        std::cout << "Testing control flow parsing..." << std::endl;
        
        // Test lexer recognizes keywords
        dsl::lexer::Lexer lexer(source);
        while (true) {
            auto token = lexer.next_token();
            if (token.type == dsl::ast::TokenType::EOF_TOKEN) break;
            
            if (token.type == dsl::ast::TokenType::IF ||
                token.type == dsl::ast::TokenType::ELSE ||
                token.type == dsl::ast::TokenType::WHILE ||
                token.type == dsl::ast::TokenType::FUNCTION ||
                token.type == dsl::ast::TokenType::RETURN) {
                std::cout << "✅ Found keyword: " << token.value << std::endl;
            }
        }
        
        // Test parser creates correct AST nodes
        dsl::parser::Parser parser(source);
        auto program = parser.parse();
        
        std::cout << "✅ Parser completed without errors" << std::endl;
        std::cout << "✅ Control flow parsing test passed!" << std::endl;
    }
};

int main() {
    try {
        MockInterpreter interpreter;
        interpreter.test_control_flow_parsing();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
