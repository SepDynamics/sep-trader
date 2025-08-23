#include "util/compiler.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "util/core_primitives.h"
#include "util/stdlib.h"
#include "util/nodes.h"

namespace dsl::compiler {

CompiledProgram Compiler::compile(const ast::Program& program) {
    std::vector<std::function<void(Context&)>> compiled_statements;
    
    // Compile all declarations
    for (const auto& stream : program.streams) {
        compiled_statements.push_back(compile_stream_declaration(*stream));
    }
    
    for (const auto& pattern : program.patterns) {
        compiled_statements.push_back(compile_pattern_declaration(*pattern));
    }
    
    for (const auto& signal : program.signals) {
        compiled_statements.push_back(compile_signal_declaration(*signal));
    }
    
    if (program.memory) {
        compiled_statements.push_back(compile_memory_declaration(*program.memory));
    }
    
    return CompiledProgram([compiled_statements](Context& context) {
        // Register built-in functions
        Compiler compiler;
        compiler.register_builtin_functions(context);
        
        // Execute all compiled statements
        for (const auto& stmt : compiled_statements) {
            stmt(context);
        }
    });
}

std::function<void(Context&)> Compiler::compile_stream_declaration(const ast::StreamDecl& stream) {
    return [stream](Context& context) {
        (void)context; // Suppress unused parameter warning
        std::cout << "Creating stream: " << stream.name << " from " << stream.source << std::endl;
        throw std::runtime_error("Stream creation requires production implementation with real data source");
    };
}

std::function<void(Context&)> Compiler::compile_pattern_declaration(const ast::PatternDecl& pattern) {
    // For simplicity, treat pattern body as a single statement
    return [pattern](Context& context) {
        std::cout << "Creating pattern: " << pattern.name << " with body: " << pattern.body << std::endl;
        context.set_variable(pattern.name, Value(std::string("pattern_active")));
        std::cout << "Pattern " << pattern.name << " execution completed" << std::endl;
    };
}

std::function<void(Context&)> Compiler::compile_signal_declaration(const ast::SignalDecl& signal) {
    // Simplified signal compilation - no expression compilation for now
    return [signal](Context& context) {
        std::cout << "Evaluating signal: " << signal.name << std::endl;
        std::cout << "Trigger condition: " << signal.trigger << std::endl;
        std::cout << "Confidence level: " << signal.confidence << std::endl;
        std::cout << "Action: " << signal.action << std::endl;
        
        // For now, just set a simple variable indicating signal is active
        context.set_variable(signal.name + "_active", Value(true));
    };
}

std::function<void(Context&)> Compiler::compile_memory_declaration(const ast::MemoryDecl& memory) {
    std::vector<std::function<void(Context&)>> compiled_rules;

    for (const auto& rule : memory.rules) {
        compiled_rules.push_back(compile_statement(*rule));
    }

    return [compiled_rules](Context& context) {
        for (const auto& rule : compiled_rules) {
            rule(context);
        }
    };
}

std::function<void(Context&)> Compiler::compile_statement(const dsl::ast::Statement& stmt) {
    if (const auto* assign = dynamic_cast<const dsl::ast::Assignment*>(&stmt)) {
        auto compiled_value = compile_expression(*assign->value);
        std::string name = assign->name;
        return [compiled_value, name](Context& context) {
            context.set_variable(name, compiled_value(context));
        };
    }
    if (const auto* expr_stmt = dynamic_cast<const dsl::ast::ExpressionStatement*>(&stmt)) {
        auto compiled_expr = compile_expression(*expr_stmt->expression);
        return [compiled_expr](Context& context) {
            compiled_expr(context);
        };
    }
    return [](Context&) {};
}

std::function<Value(Context&)> Compiler::compile_expression(const dsl::ast::Expression& expr) {
    if (auto number = dynamic_cast<const dsl::ast::NumberLiteral*>(&expr)) {
        // Capture by value to avoid use-after-free
        double value = number->value;
        return [value](Context&) -> Value {
            return Value(value);
        };
    }
    
    if (auto string = dynamic_cast<const dsl::ast::StringLiteral*>(&expr)) {
        // Capture by value to avoid use-after-free
        std::string value = string->value;
        return [value](Context&) -> Value {
            return Value(value);
        };
    }
    
    if (auto boolean = dynamic_cast<const dsl::ast::BooleanLiteral*>(&expr)) {
        bool value = boolean->value;
        return [value](Context&) -> Value {
            return Value(value);
        };
    }

    if (auto id = dynamic_cast<const dsl::ast::Identifier*>(&expr)) {
        std::string name = id->name;
        return [name](Context& context) -> Value {
            return context.get_variable(name);
        };
    }

    if (auto binop = dynamic_cast<const dsl::ast::BinaryOp*>(&expr)) {
        auto left = compile_expression(*binop->left);
        auto right = compile_expression(*binop->right);
        std::string op = binop->op;
        
        return [left, right, op](Context& context) -> Value {
            Value l = left(context);
            Value r = right(context);
            
            if (op == "+") {
                if (std::holds_alternative<double>(l) && std::holds_alternative<double>(r)) {
                    return Value(std::get<double>(l) + std::get<double>(r));
                }
            }
            // Add more operators as needed
            return Value(); // Default/error case
        };
    }
    
    if (auto identifier = dynamic_cast<const dsl::ast::Identifier*>(&expr)) {
        // Capture by value to avoid use-after-free
        std::string name = identifier->name;
        return [name](Context& context) -> Value {
            return context.get_variable(name);
        };
    }
    
    if (auto binary = dynamic_cast<const dsl::ast::BinaryOp*>(&expr)) {
        auto compiled_left = compile_expression(*binary->left);
        auto compiled_right = compile_expression(*binary->right);
        
        // Capture operator by value to avoid use-after-free
        std::string op = binary->op;
        
        return [compiled_left, compiled_right, op](Context& context) -> Value {
            Value left = compiled_left(context);
            Value right = compiled_right(context);
            
            if (op == "+") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) + std::get<double>(right));
                }
                throw std::runtime_error("Type mismatch in addition");
            } else if (op == "-") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) - std::get<double>(right));
                }
                throw std::runtime_error("Type mismatch in subtraction");
            } else if (op == "*") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) * std::get<double>(right));
                }
                throw std::runtime_error("Type mismatch in multiplication");
            } else if (op == "/") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    double right_val = std::get<double>(right);
                    if (right_val == 0.0) {
                        throw std::runtime_error("Division by zero");
                    }
                    return Value(std::get<double>(left) / right_val);
                }
                throw std::runtime_error("Type mismatch in division");
            } else if (op == ">") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) > std::get<double>(right));
                }
                throw std::runtime_error("Type mismatch in comparison");
            } else if (op == "<") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) < std::get<double>(right));
                }
                throw std::runtime_error("Type mismatch in comparison");
            } else if (op == ">=") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) >= std::get<double>(right));
                }
                throw std::runtime_error("Type mismatch in comparison");
            } else if (op == "<=") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) <= std::get<double>(right));
                }
                throw std::runtime_error("Type mismatch in comparison");
            } else if (op == "==") {
                if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                    return Value(std::get<double>(left) == std::get<double>(right));
                } else if (std::holds_alternative<bool>(left) && std::holds_alternative<bool>(right)) {
                    return Value(std::get<bool>(left) == std::get<bool>(right));
                }
                throw std::runtime_error("Type mismatch in equality");
            } else if (op == "&&") {
                if (std::holds_alternative<bool>(left) && std::holds_alternative<bool>(right)) {
                    return Value(std::get<bool>(left) && std::get<bool>(right));
                }
                throw std::runtime_error("Type mismatch in logical AND");
            } else if (op == "||") {
                if (std::holds_alternative<bool>(left) && std::holds_alternative<bool>(right)) {
                    return Value(std::get<bool>(left) || std::get<bool>(right));
                }
                throw std::runtime_error("Type mismatch in logical OR");
            }
            
            throw std::runtime_error("Unknown binary operator: " + op);
        };
    }
    
    if (auto call = dynamic_cast<const dsl::ast::Call*>(&expr)) {
        std::vector<std::function<Value(Context&)>> compiled_args;
        for (const auto& arg : call->args) {
            compiled_args.push_back(compile_expression(*arg));
        }
        
        // Capture function name by value to avoid use-after-free
        std::string function_name = call->callee;
        
        return [function_name, compiled_args](Context& context) -> Value {
            std::vector<Value> args;
            for (const auto& compiled_arg : compiled_args) {
                args.push_back(compiled_arg(context));
            }
            
            auto func = context.get_function(function_name);
            if (!func) {
                throw std::runtime_error("Unknown function: " + function_name);
            }
            return func(args);
        };
    }
    
    if (auto member = dynamic_cast<const dsl::ast::MemberAccess*>(&expr)) {
        auto compiled_object = compile_expression(*member->object);
        std::string member_name = member->member;
        
        return [compiled_object, member_name](Context& context) -> Value {
            Value object = compiled_object(context);
            
            // For pattern member access, construct the variable name
            if (std::holds_alternative<std::string>(object)) {
                std::string pattern_name = std::get<std::string>(object);
                std::string full_var_name = pattern_name + "." + member_name;
                return context.get_variable(full_var_name);
            }
            
            return Value{}; // Fallback for unsupported member access
        };
    }

    // Default case
    return [](Context&) -> Value { return Value{}; };
}

void Compiler::register_builtin_functions(Context& context) {
    // Register all standard library modules
    // Note: Context type mismatch - sep_dsl::stdlib expects VMExecutionContext
    // For now, only register compatible functions
    try {
        dsl::stdlib::register_core_primitives(context);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to register some builtin functions: " << e.what() << std::endl;
    }
}

} // namespace dsl::compiler
