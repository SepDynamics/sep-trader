#include "dsl/parser/parser.h"
#include "dsl/runtime/interpreter.h"
#include <fstream>
#include <iostream>
#include <streambuf>

int main() {
    std::ifstream t("examples/weighted_sum_test.dsl");
    std::string str((std::istreambuf_iterator<char>(t)),
                    std::istreambuf_iterator<char>());

    dsl::parser::Parser parser(str);
    auto program = parser.parse();

    dsl::runtime::Interpreter interpreter;
    
    // Manually set input values for testing
    interpreter.interpret(*program);
    
    // TODO: Add a way to get the result of the pattern and assert on it
    
    return 0;
}