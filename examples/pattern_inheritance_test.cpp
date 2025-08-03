#include "dsl/lexer/lexer.h"
#include "dsl/parser/parser.h"
#include "dsl/runtime/interpreter.h"
#include <fstream>
#include <iostream>
#include <sstream>

int main() {
    std::ifstream file("examples/pattern_inheritance_test.dsl");
    if (!file) {
        std::cerr << "Could not open file." << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    try {
        dsl::parser::Parser parser(source);
        auto program = parser.parse();

        dsl::runtime::Interpreter interpreter;
        interpreter.interpret(*program);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}