#include <iostream>
#include <string>
#include <vector>
#include "dsl/parser/parser.h"
#include "dsl/lexer/lexer.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    std::string input(reinterpret_cast<const char*>(Data), Size);
    Lexer lexer(input);
    Parser parser(lexer);
    try {
        parser.parse();
    } catch (const std::exception& e) {
        // Ignore exceptions
    }
    return 0;
}
