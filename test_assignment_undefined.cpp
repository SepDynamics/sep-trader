#include "src/dsl/parser/parser.h"
#include "src/dsl/runtime/interpreter.h"
#include <iostream>
#include <stdexcept>

using namespace dsl;

int main() {
    try {
        Parser parser("result = undefined_variable + 5");
        auto program = parser.parse();
        if (!program) {
            std::cout << "Parse failed\n";
            return 1;
        }
        
        Interpreter interpreter;
        interpreter.interpret(*program);
        std::cout << "No exception thrown!\n";
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << "\n";
    }
    return 0;
}
