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
        std::cout << "Creating stream: " << stream.name << " from " << stream.source << std::endl;

#ifdef SEP_BACKTESTING
        // Backtesting placeholder: attach mock stream data
        context.set_variable(stream.name, Value(std::string("stream_data")));
#else
        throw std::runtime_error("Stream creation requires production implementation");
#endif
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
        // Create a Statement object from the rule string
        ast::Statement stmt;
        stmt.content = rule;
        compiled_rules.push_back(compile_statement(stmt));
    }
    
    return [compiled_rules](Context& context) {
        std::cout << "Executing memory rules" << std::endl;
        
        for (const auto& rule : compiled_rules) {
            rule(context);
        }
    };
}

std::function<void(Context&)> Compiler::compile_statement(const ast::Statement& stmt) {
    // Simplified statement compilation - just execute the content as a placeholder
    return [stmt](Context& context) {
        std::cout << "Executing statement: " << stmt.content << std::endl;
        // For now, just set a simple variable to indicate the statement ran
        context.set_variable("last_statement", Value(stmt.content));
    };
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
    
    if (auto identifier = dynamic_cast<const ast::IdentifierExpr*>(&expr)) {
        // Capture by value to avoid use-after-free
        std::string name = identifier->name;
        return [name](Context& context) -> Value {
            return context.get_variable(name);
        };
    }
    
    if (auto binary = dynamic_cast<const ast::BinaryExpr*>(&expr)) {
        auto compiled_left = compile_expression(*binary->left);
        auto compiled_right = compile_expression(*binary->right);
        
        // Capture operator by value to avoid use-after-free
        ast::TokenType op_type = binary->operator_type;
        
        return [compiled_left, compiled_right, op_type](Context& context) -> Value {
            Value left = compiled_left(context);
            Value right = compiled_right(context);
            
            switch (op_type) {
                case ast::TokenType::PLUS:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() + right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in addition");
                    break;
                case ast::TokenType::MINUS:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() - right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in subtraction");
                case ast::TokenType::MULTIPLY:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() * right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in multiplication");
                case ast::TokenType::DIVIDE:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() / right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in division");
                case ast::TokenType::GT:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() > right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in comparison");
                case ast::TokenType::LT:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() < right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in comparison");
                case ast::TokenType::GE:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() >= right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in comparison");
                case ast::TokenType::LE:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() <= right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in comparison");
                case ast::TokenType::EQ:
                    if (left.type == Value::NUMBER && right.type == Value::NUMBER) {
                        return Value(left.get<double>() == right.get<double>());
                    }
                    throw std::runtime_error("Type mismatch in equality");
                case ast::TokenType::AND:
                    if (left.type == Value::BOOLEAN && right.type == Value::BOOLEAN) {
                        return Value(left.get<bool>() && right.get<bool>());
                    }
                    throw std::runtime_error("Type mismatch in logical AND");
                case ast::TokenType::OR:
                    if (left.type == Value::BOOLEAN && right.type == Value::BOOLEAN) {
                        return Value(left.get<bool>() || right.get<bool>());
                    }
                    throw std::runtime_error("Type mismatch in logical OR");
            }
            
            throw std::runtime_error("Invalid binary operation");
        };
    }
    
    if (auto call = dynamic_cast<const ast::CallExpr*>(&expr)) {
        std::vector<std::function<Value(Context&)>> compiled_args;
        for (const auto& arg : call->arguments) {
            compiled_args.push_back(compile_expression(*arg));
        }
        
        // Capture function name by value to avoid use-after-free
        std::string function_name = call->function_name;
        
        return [function_name, compiled_args](Context& context) -> Value {
            std::vector<Value> args;
            for (const auto& compiled_arg : compiled_args) {
                args.push_back(compiled_arg(context));
            }
            
            return context.call_function(function_name, args);
        };
    }
    
    if (auto member = dynamic_cast<const ast::MemberExpr*>(&expr)) {
        auto compiled_object = compile_expression(*member->object);
        std::string member_name = member->member;
        
        return [compiled_object, member_name](Context& context) -> Value {
            Value object = compiled_object(context);
            
            // For pattern member access, construct the variable name
            if (object.type == Value::STRING) {
                std::string pattern_name = object.get<std::string>();
                std::string full_var_name = pattern_name + "." + member_name;
                return context.get_variable(full_var_name);
            }
            
            return Value(0.5); // Fallback
        };
    }
    
    // Default case
    return [](Context& context) -> Value {
        return Value(0.0);
    };
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
