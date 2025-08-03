#include "runtime.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace dsl::runtime {

DSLRuntime::DSLRuntime() : current_parser(nullptr) {}

DSLRuntime::~DSLRuntime() {
    delete current_parser;
}

void DSLRuntime::execute(const std::string& source) {
    try {
        // Parse the source code
        parser::Parser parser(source);
        auto program = parser.parse();
        
        // Execute using the interpreter
        interpreter_.interpret(*program);
        
    } catch (const std::exception& e) {
        handle_error("Execution error: " + std::string(e.what()));
    }
}

void DSLRuntime::execute_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        handle_error("Cannot open file: " + filename);
        return;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    execute(buffer.str());
}

void DSLRuntime::start_repl() {
    std::cout << "DSL REPL - Enter commands (type 'exit' to quit):" << std::endl;
    std::string line;
    
    while (true) {
        std::cout << "dsl> ";
        if (!std::getline(std::cin, line)) {
            break;
        }
        
        if (line == "exit" || line == "quit") {
            break;
        }
        
        if (!line.empty()) {
            execute_line(line);
        }
    }
}

void DSLRuntime::execute_line(const std::string& line) {
    execute(line);
}

void DSLRuntime::set_error_handler(std::function<void(const std::string&)> handler) {
    error_handler = handler;
}

void DSLRuntime::handle_error(const std::string& message) {
    if (error_handler) {
        error_handler(message);
    } else {
        std::cerr << "Error: " << message << std::endl;
    }
}

// Variable access methods
Value DSLRuntime::get_variable(const std::string& name) {
    return interpreter_.get_global_variable(name);
}

bool DSLRuntime::has_variable(const std::string& name) {
    return interpreter_.has_global_variable(name);
}

std::unordered_map<std::string, Value> DSLRuntime::get_all_variables() {
    return interpreter_.get_global_variables();
}

} // namespace dsl::runtime
