#include "lexer/lexer.h"
#include "parser/parser.h"
#include "runtime/interpreter.h"
#include "engine/facade/facade.h"
#include "core_types/result.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <script.sep>" << std::endl;
        return 1;
    }
    
    // Read the DSL script file
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << argv[1] << std::endl;
        return 1;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    file.close();
    
    try {
        std::cout << "=== SEP DSL Interpreter ===" << std::endl;
        std::cout << "Source file: " << argv[1] << std::endl;
        std::cout << "Source content:" << std::endl;
        std::cout << source << std::endl;
        std::cout << "========================" << std::endl;
        
        // Create parser and parse the source
        dsl::parser::Parser parser(source);
        auto program = parser.parse();
        
        std::cout << "Parsing completed successfully!" << std::endl;
        std::cout << "Program contains:" << std::endl;
        std::cout << "  - " << program->streams.size() << " stream(s)" << std::endl;
        std::cout << "  - " << program->patterns.size() << " pattern(s)" << std::endl;
        std::cout << "  - " << program->signals.size() << " signal(s)" << std::endl;
        std::cout << "========================" << std::endl;
        
        // Initialize the engine facade before creating interpreter
        auto& engine = sep::engine::EngineFacade::getInstance();
        auto init_result = engine.initialize();
        if (!sep::core::isSuccess(init_result)) {
            std::cerr << "Failed to initialize engine facade" << std::endl;
            return 1;
        }
        
        // Create interpreter and execute the program
        dsl::runtime::Interpreter interpreter;
        std::cout << "Starting interpretation:" << std::endl;
        interpreter.interpret(*program);
        
        std::cout << "========================" << std::endl;
        std::cout << "Interpretation completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
