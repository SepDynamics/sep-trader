#include "src/dsl/lexer/lexer.h"
#include "src/dsl/parser/parser.h"
#include "src/dsl/runtime/interpreter.h"
#include <iostream>

int main() {
    try {
        std::string source = R"(
            // Test basic control flow
            x = 5
            
            if (x > 3) {
                result = "greater"
            } else {
                result = "smaller"
            }
            
            // Test while loop
            counter = 0
            while (counter < 3) {
                counter = counter + 1
            }
        )";
        
        std::cout << "Parsing DSL source..." << std::endl;
        dsl::parser::Parser parser(source);
        auto program = parser.parse();
        
        std::cout << "Interpreting..." << std::endl;
        dsl::runtime::Interpreter interpreter;
        interpreter.interpret(*program);
        
        std::cout << "DSL execution completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
