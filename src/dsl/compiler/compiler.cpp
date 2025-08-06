#include "compiler.h"

#include <cmath>
#include <iostream>

#include "stdlib/stdlib.h"

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
        
        // Create a mock stream value for now
        context.set_variable(stream.name, Value("stream_data"));
    };
}

std::function<void(Context&)> Compiler::compile_pattern_declaration(const ast::PatternDecl& pattern) {
    std::vector<std::function<void(Context&)>> compiled_body;
    
    for (const auto& stmt : pattern.body) {
        compiled_body.push_back(compile_statement(*stmt));
    }
    
    // Capture by value to avoid copy issues
    std::string pattern_name = pattern.name;
    
    return [pattern_name, compiled_body](Context& context) {
        std::cout << "Executing pattern: " << pattern_name << std::endl;
        
        // Set up pattern context and identifier
        context.set_variable("_current_pattern", Value(pattern_name));
        context.set_variable(pattern_name, Value(pattern_name));
        
        // Execute pattern body with safety checks
        try {
            for (size_t i = 0; i < compiled_body.size(); ++i) {
                std::cout << "  Executing statement " << i << std::endl;
                compiled_body[i](context);
                std::cout << "  Statement " << i << " completed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Error executing pattern body: " << e.what() << std::endl;
            throw;
        }
        
        // Clear pattern context but keep pattern identifier for member access
        context.set_variable("_current_pattern", Value(""));
        std::cout << "Pattern " << pattern_name << " execution completed" << std::endl;
    };
}

std::function<void(Context&)> Compiler::compile_signal_declaration(const ast::SignalDecl& signal) {
    auto compiled_trigger = signal.trigger ? compile_expression(*signal.trigger) : nullptr;
    auto compiled_confidence = signal.confidence ? compile_expression(*signal.confidence) : nullptr;
    
    // Capture by value only what we need to avoid copy issues
    std::string signal_name = signal.name;
    std::string signal_action = signal.action;
    
    return [signal_name, signal_action, compiled_trigger, compiled_confidence](Context& context) {
        std::cout << "Evaluating signal: " << signal_name << std::endl;
        
        if (compiled_trigger) {
            Value trigger_result = compiled_trigger(context);
            bool should_trigger = false;
            if (trigger_result.type == Value::BOOLEAN) {
                should_trigger = trigger_result.get<bool>();
            } else if (trigger_result.type == Value::NUMBER) {
                should_trigger = trigger_result.get<double>() != 0.0;
            }
            
            if (should_trigger) {
                std::cout << "Signal triggered! Action: " << signal_action << std::endl;
                
                if (compiled_confidence) {
                    Value confidence_value = compiled_confidence(context);
                    std::cout << "Confidence: " << confidence_value.get<double>() << std::endl;
                }
            }
        }
    };
}

std::function<void(Context&)> Compiler::compile_memory_declaration(const ast::MemoryDecl& memory) {
    std::vector<std::function<void(Context&)>> compiled_rules;
    
    for (const auto& rule : memory.rules) {
        compiled_rules.push_back(compile_statement(*rule));
    }
    
    return [compiled_rules](Context& context) {
        std::cout << "Executing memory rules" << std::endl;
        
        for (const auto& rule : compiled_rules) {
            rule(context);
        }
    };
}

std::function<void(Context&)> Compiler::compile_statement(const ast::Statement& stmt) {
    if (auto assignment = dynamic_cast<const ast::AssignmentStmt*>(&stmt)) {
        auto compiled_value = compile_expression(*assignment->value);
        
        // Capture by value to avoid use-after-free
        std::string variable_name = assignment->variable;
        
        return [variable_name, compiled_value](Context& context) {
            Value result = compiled_value(context);
            context.set_variable(variable_name, result);
            
            // Also store with pattern prefix for member access
            // Check if we're in a pattern context (look for _current_pattern)
            try {
                Value current_pattern = context.get_variable("_current_pattern");
                if (current_pattern.type == Value::STRING) {
                    std::string pattern_name = current_pattern.get<std::string>();
                    std::string full_name = pattern_name + "." + variable_name;
                    context.set_variable(full_name, result);
                }
            } catch (...) {
                // No current pattern context
            }
            
            std::cout << "Assigned " << variable_name << " = ";
            
            switch (result.type) {
                case Value::NUMBER:
                    std::cout << result.get<double>() << std::endl;
                    break;
                case Value::STRING:
                    std::cout << "\"" << result.get<std::string>() << "\"" << std::endl;
                    break;
                case Value::BOOLEAN:
                    std::cout << (result.get<bool>() ? "true" : "false") << std::endl;
                    break;
                default:
                    std::cout << "[complex value]" << std::endl;
            }
        };
    }
    
    if (auto expr_stmt = dynamic_cast<const ast::ExpressionStmt*>(&stmt)) {
        auto compiled_expr = compile_expression(*expr_stmt->expression);
        
        return [compiled_expr](Context& context) {
            compiled_expr(context);
        };
    }
    
    if (auto if_stmt = dynamic_cast<const ast::IfStmt*>(&stmt)) {
        auto compiled_condition = compile_expression(*if_stmt->condition);
        
        std::vector<std::function<void(Context&)>> compiled_then;
        for (const auto& then_stmt : if_stmt->then_block) {
            compiled_then.push_back(compile_statement(*then_stmt));
        }
        
        std::vector<std::function<void(Context&)>> compiled_else;
        for (const auto& else_stmt : if_stmt->else_block) {
            compiled_else.push_back(compile_statement(*else_stmt));
        }
        
        return [compiled_condition, compiled_then, compiled_else](Context& context) {
            Value condition_result = compiled_condition(context);
            
            if (condition_result.type == Value::BOOLEAN && condition_result.get<bool>()) {
                for (const auto& stmt : compiled_then) {
                    stmt(context);
                }
            } else {
                for (const auto& stmt : compiled_else) {
                    stmt(context);
                }
            }
        };
    }
    
    // Default case
    return [](Context& context) {
        std::cout << "Unknown statement executed" << std::endl;
    };
}

std::function<Value(Context&)> Compiler::compile_expression(const ast::Expression& expr) {
    if (auto number = dynamic_cast<const ast::NumberExpr*>(&expr)) {
        // Capture by value to avoid use-after-free
        double value = number->value;
        return [value](Context&) -> Value {
            return Value(value);
        };
    }
    
    if (auto string = dynamic_cast<const ast::StringExpr*>(&expr)) {
        // Capture by value to avoid use-after-free
        std::string value = string->value;
        return [value](Context&) -> Value {
            return Value(value);
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
    stdlib::register_all(context);
    
    // Legacy functions for backwards compatibility
    context.set_function("qfh", [this](const std::vector<Value>& args) { return builtin_qfh(args); });
    context.set_function("qbsa", [this](const std::vector<Value>& args) { return builtin_qbsa(args); });
    context.set_function("coherence", [this](const std::vector<Value>& args) { return builtin_coherence(args); });
    context.set_function("stability", [this](const std::vector<Value>& args) { return builtin_stability(args); });
    context.set_function("entropy", [this](const std::vector<Value>& args) { return builtin_entropy(args); });
}

Value Compiler::builtin_qfh(const std::vector<Value>& args) {
    std::cout << "Executing QFH analysis..." << std::endl;
    // Mock QFH computation
    return Value(0.75); // Mock coherence result
}

Value Compiler::builtin_qbsa(const std::vector<Value>& args) {
    std::cout << "Executing QBSA analysis..." << std::endl;
    // Mock QBSA computation
    return Value(0.68); // Mock stability result
}

Value Compiler::builtin_coherence(const std::vector<Value>& args) {
    std::cout << "Computing coherence..." << std::endl;
    // Mock coherence computation
    return Value(0.82);
}

Value Compiler::builtin_stability(const std::vector<Value>& args) {
    std::cout << "Computing stability..." << std::endl;
    // Mock stability computation
    return Value(0.73);
}

Value Compiler::builtin_entropy(const std::vector<Value>& args) {
    std::cout << "Computing entropy..." << std::endl;
    // Mock entropy computation
    return Value(0.45);
}

Value Compiler::builtin_weighted_sum(const std::vector<Value>& args) {
    std::cout << "Computing weighted sum..." << std::endl;
    // Mock weighted sum - in real implementation, this would take weight parameters
    return Value(0.77);
}

} // namespace dsl::compiler
