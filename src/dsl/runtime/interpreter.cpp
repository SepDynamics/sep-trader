#include "interpreter.h"
#include "engine/facade/facade.h"
#include <iostream>
#include <stdexcept>

namespace dsl::runtime {

// Environment implementation
void Environment::define(const std::string& name, const Value& value) {
    variables_[name] = value;
}

Value Environment::get(const std::string& name) {
    if (variables_.find(name) != variables_.end()) {
        return variables_[name];
    }
    
    if (enclosing_ != nullptr) {
        return enclosing_->get(name);
    }
    
    throw std::runtime_error("Undefined variable '" + name + "'.");
}

void Environment::assign(const std::string& name, const Value& value) {
    if (variables_.find(name) != variables_.end()) {
        variables_[name] = value;
        return;
    }
    
    if (enclosing_ != nullptr) {
        enclosing_->assign(name, value);
        return;
    }
    
    throw std::runtime_error("Undefined variable '" + name + "'.");
}

// Interpreter implementation
void Interpreter::interpret(const ast::Program& program) {
    environment_ = &globals_;
    
    try {
        // Execute stream declarations
        for (const auto& stream : program.streams) {
            execute_stream_decl(*stream);
        }
        
        // Execute pattern declarations
        for (const auto& pattern : program.patterns) {
            execute_pattern_decl(*pattern);
        }
        
        // Execute signal declarations
        for (const auto& signal : program.signals) {
            execute_signal_decl(*signal);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
    }
}

Value Interpreter::evaluate(const ast::Expression& expr) {
    // Use dynamic casting to determine the actual type
    if (const auto* number = dynamic_cast<const ast::NumberLiteral*>(&expr)) {
        return visit_number_literal(*number);
    }
    if (const auto* string_lit = dynamic_cast<const ast::StringLiteral*>(&expr)) {
        return visit_string_literal(*string_lit);
    }
    if (const auto* identifier = dynamic_cast<const ast::Identifier*>(&expr)) {
        return visit_identifier(*identifier);
    }
    if (const auto* binary_op = dynamic_cast<const ast::BinaryOp*>(&expr)) {
        return visit_binary_op(*binary_op);
    }
    if (const auto* call = dynamic_cast<const ast::Call*>(&expr)) {
        return visit_call(*call);
    }
    if (const auto* member_access = dynamic_cast<const ast::MemberAccess*>(&expr)) {
        return visit_member_access(*member_access);
    }
    
    throw std::runtime_error("Unknown expression type");
}

void Interpreter::execute(const ast::Statement& stmt) {
    if (const auto* assignment = dynamic_cast<const ast::Assignment*>(&stmt)) {
        visit_assignment(*assignment);
    } else if (const auto* expr_stmt = dynamic_cast<const ast::ExpressionStatement*>(&stmt)) {
        visit_expression_statement(*expr_stmt);
    } else {
        throw std::runtime_error("Unknown statement type");
    }
}

Value Interpreter::visit_number_literal(const ast::NumberLiteral& node) {
    return node.value;
}

Value Interpreter::visit_string_literal(const ast::StringLiteral& node) {
    return node.value;
}

Value Interpreter::visit_identifier(const ast::Identifier& node) {
    return environment_->get(node.name);
}

Value Interpreter::visit_binary_op(const ast::BinaryOp& node) {
    Value left = evaluate(*node.left);
    Value right = evaluate(*node.right);
    
    if (node.op == "+") {
        // Handle both numeric and string concatenation
        try {
            double left_num = std::any_cast<double>(left);
            double right_num = std::any_cast<double>(right);
            return left_num + right_num;
        } catch (const std::bad_any_cast&) {
            std::string left_str = stringify(left);
            std::string right_str = stringify(right);
            return left_str + right_str;
        }
    }
    if (node.op == "-") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num - right_num;
    }
    if (node.op == "*") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num * right_num;
    }
    if (node.op == "/") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        if (right_num == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        return left_num / right_num;
    }
    if (node.op == ">") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num > right_num;
    }
    if (node.op == "<") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num < right_num;
    }
    if (node.op == ">=") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num >= right_num;
    }
    if (node.op == "<=") {
        double left_num = std::any_cast<double>(left);
        double right_num = std::any_cast<double>(right);
        return left_num <= right_num;
    }
    if (node.op == "==") {
        return is_equal(left, right);
    }
    if (node.op == "!=") {
        return !is_equal(left, right);
    }
    
    throw std::runtime_error("Unknown binary operator: " + node.op);
}

Value Interpreter::visit_call(const ast::Call& node) {
    std::vector<Value> arguments;
    for (const auto& arg : node.args) {
        arguments.push_back(evaluate(*arg));
    }
    
    return call_builtin_function(node.callee, arguments);
}

Value Interpreter::visit_member_access(const ast::MemberAccess& node) {
    Value object = evaluate(*node.object);
    std::cout << "Member access: looking for member '" << node.member << "'" << std::endl;
    
    try {
        // Try to cast object to PatternResult
        PatternResult pattern_result = std::any_cast<PatternResult>(object);
        std::cout << "Found pattern result with " << pattern_result.size() << " members" << std::endl;
        
        // Debug: print all available members
        for (const auto& [name, value] : pattern_result) {
            std::cout << "  Available member: " << name << std::endl;
        }
        
        // Look up the member in the pattern result
        auto it = pattern_result.find(node.member);
        if (it != pattern_result.end()) {
            std::cout << "Found member '" << node.member << "'" << std::endl;
            return it->second;
        } else {
            throw std::runtime_error("Pattern member not found: " + node.member);
        }
    } catch (const std::bad_any_cast&) {
        throw std::runtime_error("Cannot access member of non-pattern object");
    }
}

void Interpreter::visit_assignment(const ast::Assignment& node) {
    Value value = evaluate(*node.value);
    environment_->define(node.name, value);
}

void Interpreter::visit_expression_statement(const ast::ExpressionStatement& node) {
    Value result = evaluate(*node.expression);
    // For expression statements, we might want to print the result
    std::cout << stringify(result) << std::endl;
}

void Interpreter::execute_stream_decl(const ast::StreamDecl& decl) {
    std::cout << "Executing stream declaration: " << decl.name << " from " << decl.source << std::endl;
    
    // For now, just store the stream info in the environment
    environment_->define(decl.name + "_source", decl.source);
    
    // Store parameters
    for (const auto& param : decl.params) {
        environment_->define(decl.name + "_" + param.first, param.second);
    }
}

void Interpreter::execute_pattern_decl(const ast::PatternDecl& decl) {
    std::cout << "Executing pattern declaration: " << decl.name << std::endl;
    
    // Create a new environment for the pattern
    Environment pattern_env(environment_);
    Environment* previous = environment_;
    environment_ = &pattern_env;
    
    // Define inputs in the pattern environment
    for (const auto& input : decl.inputs) {
        environment_->define(input, std::string("input_placeholder"));
    }
    
    // Execute pattern body
    for (const auto& stmt : decl.body) {
        execute(*stmt);
    }
    
    // Capture pattern results from the pattern environment
    PatternResult pattern_result;
    // Copy all variables from pattern_env to pattern_result
    // Note: This is a simplified approach - in practice you'd want to be more selective
    try {
        // Get all variables defined in this pattern scope
        const auto& pattern_vars = pattern_env.getVariables();
        for (const auto& [name, value] : pattern_vars) {
            pattern_result[name] = value;
        }
        
        // Store the pattern result in global environment under the pattern name
        globals_.define(decl.name, pattern_result);
        std::cout << "Pattern " << decl.name << " captured " << pattern_result.size() << " variables" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Warning: Could not capture pattern results: " << e.what() << std::endl;
    }
    
    // Restore previous environment
    environment_ = previous;
}

void Interpreter::execute_signal_decl(const ast::SignalDecl& decl) {
    std::cout << "Executing signal declaration: " << decl.name << std::endl;
    
    if (decl.trigger) {
        Value trigger_result = evaluate(*decl.trigger);
        bool should_trigger = is_truthy(trigger_result);
        
        if (should_trigger) {
            std::cout << "Signal " << decl.name << " triggered! Action: " << decl.action << std::endl;
            
            if (decl.confidence) {
                Value confidence_value = evaluate(*decl.confidence);
                std::cout << "Confidence: " << stringify(confidence_value) << std::endl;
            }
        }
    }
}

Value Interpreter::call_builtin_function(const std::string& name, const std::vector<Value>& args) {
    // Get the singleton instance of the engine facade
    auto& engine = sep::engine::EngineFacade::getInstance();
    
    if (name == "measure_coherence") {
        std::cout << "Calling real measure_coherence with " << args.size() << " arguments" << std::endl;
        
        // Convert DSL arguments into the request struct
        sep::engine::PatternAnalysisRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "default_pattern";
            }
        }
        request.analysis_depth = 3;
        request.include_relationships = true;
        
        // Call the real C++ function
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            return static_cast<double>(response.confidence_score);
        } else {
            std::cout << "Engine call failed, returning mock value" << std::endl;
            return 0.82; // Fallback to mock
        }
    }
    
    if (name == "qfh_analyze") {
        std::cout << "Calling qfh_analyze with " << args.size() << " arguments" << std::endl;
        // QFH processor is lower-level - keeping mocked for now
        // Future: integrate with sep::quantum::bitspace::QFHBasedProcessor
        return 0.75; // Mock coherence value
    }
    
    if (name == "measure_entropy") {
        std::cout << "Calling measure_entropy with " << args.size() << " arguments" << std::endl;
        // Future: integrate with entropy measurement from quantum engine
        return 0.65;
    }
    
    if (name == "extract_bits") {
        std::cout << "Calling extract_bits with " << args.size() << " arguments" << std::endl;
        // Future: integrate with bitspace extraction
        return std::string("101010101");
    }
    
    throw std::runtime_error("Unknown function: " + name);
}

bool Interpreter::is_truthy(const Value& value) {
    try {
        return std::any_cast<bool>(value);
    } catch (const std::bad_any_cast&) {
        try {
            double num = std::any_cast<double>(value);
            return num != 0.0;
        } catch (const std::bad_any_cast&) {
            try {
                std::string str = std::any_cast<std::string>(value);
                return !str.empty();
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
    }
}

bool Interpreter::is_equal(const Value& a, const Value& b) {
    // Simplified equality check
    try {
        double a_num = std::any_cast<double>(a);
        double b_num = std::any_cast<double>(b);
        return a_num == b_num;
    } catch (const std::bad_any_cast&) {
        try {
            std::string a_str = std::any_cast<std::string>(a);
            std::string b_str = std::any_cast<std::string>(b);
            return a_str == b_str;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
}

std::string Interpreter::stringify(const Value& value) {
    try {
        return std::to_string(std::any_cast<double>(value));
    } catch (const std::bad_any_cast&) {
        try {
            return std::any_cast<std::string>(value);
        } catch (const std::bad_any_cast&) {
            try {
                bool b = std::any_cast<bool>(value);
                return b ? "true" : "false";
            } catch (const std::bad_any_cast&) {
                return "<unknown value>";
            }
        }
    }
}

} // namespace dsl::runtime
