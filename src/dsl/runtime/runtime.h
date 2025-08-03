#pragma once
#include "interpreter.h"
#include "parser/parser.h"
#include <string>
#include <functional>

namespace dsl::runtime {

class DSLRuntime {
private:
    Interpreter interpreter_;
    parser::Parser* current_parser;

public:
    DSLRuntime();
    ~DSLRuntime();
    
    // Execute DSL code from string
    void execute(const std::string& source);
    
    // Execute DSL code from file
    void execute_file(const std::string& filename);
    
    // Interactive REPL functions
    void start_repl();
    void execute_line(const std::string& line);
    
    // Variable access
    Value get_variable(const std::string& name);
    bool has_variable(const std::string& name);
    std::unordered_map<std::string, Value> get_all_variables();
    
    // Error handling
    void set_error_handler(std::function<void(const std::string&)> handler);
    
private:
    std::function<void(const std::string&)> error_handler;
    void handle_error(const std::string& message);
};

} // namespace dsl::runtime
