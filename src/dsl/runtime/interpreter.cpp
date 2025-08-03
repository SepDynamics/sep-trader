#include "interpreter.h"
#include "engine/facade/facade.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

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
Interpreter::Interpreter() : environment_(&globals_), program_(nullptr) {
    register_builtins();
}

void Interpreter::register_builtins() {
    // Get the singleton instance of the engine facade
    auto& engine = sep::engine::EngineFacade::getInstance();
    
    // AGI Engine Bridge Functions - THE REAL POWER
    builtins_["measure_coherence"] = [&engine](const std::vector<Value>& args) -> Value {
        std::cout << "DSL: Calling real measure_coherence with " << args.size() << " arguments" << std::endl;
        
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
        
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            return static_cast<double>(response.confidence_score);
        } else {
            throw std::runtime_error("Engine call failed for measure_coherence");
        }
    };
    
    builtins_["qfh_analyze"] = [&engine](const std::vector<Value>& args) -> Value {
        if (args.empty()) {
            throw std::runtime_error("qfh_analyze expects a bitstream argument");
        }

        std::vector<uint8_t> bitstream;
        try {
            std::string bitstream_str = std::any_cast<std::string>(args[0]);
            for (char c : bitstream_str) {
                bitstream.push_back(c - '0');
            }
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid bitstream argument for qfh_analyze");
        }

        sep::engine::QFHAnalysisRequest request;
        request.bitstream = bitstream;
        sep::engine::QFHAnalysisResponse response;
        auto result = engine.qfhAnalyze(request, response);

        if (sep::core::isSuccess(result)) {
            return static_cast<double>(response.rupture_ratio);
        } else {
            throw std::runtime_error("Engine call failed for qfh_analyze");
        }
    };
    
    builtins_["measure_entropy"] = [&engine](const std::vector<Value>& args) -> Value {
        std::cout << "DSL: Calling real measure_entropy with " << args.size() << " arguments" << std::endl;
        
        sep::engine::PatternAnalysisRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "entropy_pattern";
            }
        }
        request.analysis_depth = 2;
        request.include_relationships = false;
        
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            std::cout << "Real entropy from engine: " << response.entropy << std::endl;
            return static_cast<double>(response.entropy);
        } else {
            throw std::runtime_error("Engine call failed for measure_entropy");
        }
    };
    
    builtins_["extract_bits"] = [&engine](const std::vector<Value>& args) -> Value {
        std::cout << "DSL: Calling real extract_bits with " << args.size() << " arguments" << std::endl;
        
        sep::engine::BitExtractionRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "bitstream_pattern";
            }
        }
        
        sep::engine::BitExtractionResponse response;
        auto result = engine.extractBits(request, response);
        
        if (sep::core::isSuccess(result) && response.success) {
            // Convert bitstream to string for DSL use
            std::string bitstream_str;
            for (uint8_t bit : response.bitstream) {
                bitstream_str += (bit ? '1' : '0');
            }
            return bitstream_str;
        } else {
            throw std::runtime_error("Engine call failed for extract_bits");
        }
    };
    
    builtins_["manifold_optimize"] = [&engine](const std::vector<Value>& args) -> Value {
        if (args.size() < 3) {
            throw std::runtime_error("manifold_optimize expects pattern_id, target_coherence, target_stability");
        }
        
        sep::engine::ManifoldOptimizationRequest request;
        try {
            request.pattern_id = std::any_cast<std::string>(args[0]);
            request.target_coherence = static_cast<float>(std::any_cast<double>(args[1]));
            request.target_stability = static_cast<float>(std::any_cast<double>(args[2]));
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid arguments for manifold_optimize");
        }
        
        sep::engine::ManifoldOptimizationResponse response;
        auto result = engine.manifoldOptimize(request, response);
        
        if (sep::core::isSuccess(result) && response.success) {
            return static_cast<double>(response.optimized_coherence);
        } else {
            throw std::runtime_error("Engine call failed for manifold_optimize");
        }
    };
    
    // Math functions
    builtins_["abs"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("abs() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        return std::abs(value);
    };
    
    builtins_["sqrt"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 1) {
            throw std::runtime_error("sqrt() expects exactly 1 argument");
        }
        double value = std::any_cast<double>(args[0]);
        if (value < 0) {
            throw std::runtime_error("sqrt() of negative number");
        }
        return std::sqrt(value);
    };
    
    builtins_["min"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("min() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::min(a, b);
    };
    
    builtins_["max"] = [](const std::vector<Value>& args) -> Value {
        if (args.size() != 2) {
            throw std::runtime_error("max() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::max(a, b);
    };
    
    // Debugging functions
    builtins_["print"] = [this](const std::vector<Value>& args) -> Value {
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << stringify(args[i]);
        }
        std::cout << std::endl;
        return 0.0; // Return dummy value
    };
}

void Interpreter::interpret(const ast::Program& program) {
    environment_ = &globals_;
    program_ = &program;
    
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
    if (const auto* boolean_lit = dynamic_cast<const ast::BooleanLiteral*>(&expr)) {
        return visit_boolean_literal(*boolean_lit);
    }
    if (const auto* identifier = dynamic_cast<const ast::Identifier*>(&expr)) {
        return visit_identifier(*identifier);
    }
    if (const auto* binary_op = dynamic_cast<const ast::BinaryOp*>(&expr)) {
        return visit_binary_op(*binary_op);
    }
    if (const auto* unary_op = dynamic_cast<const ast::UnaryOp*>(&expr)) {
        return visit_unary_op(*unary_op);
    }
    if (const auto* call = dynamic_cast<const ast::Call*>(&expr)) {
        return visit_call(*call);
    }
    if (const auto* member_access = dynamic_cast<const ast::MemberAccess*>(&expr)) {
        return visit_member_access(*member_access);
    }
    if (const auto* weighted_sum = dynamic_cast<const ast::WeightedSum*>(&expr)) {
       return visit_weighted_sum(*weighted_sum);
    }
    
    throw std::runtime_error("Unknown expression type");
}

void Interpreter::execute(const ast::Statement& stmt) {
    if (const auto* assignment = dynamic_cast<const ast::Assignment*>(&stmt)) {
        visit_assignment(*assignment);
    } else if (const auto* expr_stmt = dynamic_cast<const ast::ExpressionStatement*>(&stmt)) {
        visit_expression_statement(*expr_stmt);
    } else if (const auto* evolve_stmt = dynamic_cast<const ast::EvolveStatement*>(&stmt)) {
        visit_evolve_statement(*evolve_stmt);
    } else if (const auto* if_stmt = dynamic_cast<const ast::IfStatement*>(&stmt)) {
        visit_if_statement(*if_stmt);
    } else if (const auto* while_stmt = dynamic_cast<const ast::WhileStatement*>(&stmt)) {
        visit_while_statement(*while_stmt);
    } else if (const auto* func_decl = dynamic_cast<const ast::FunctionDeclaration*>(&stmt)) {
        visit_function_declaration(*func_decl);
    } else if (const auto* return_stmt = dynamic_cast<const ast::ReturnStatement*>(&stmt)) {
        visit_return_statement(*return_stmt);
    } else if (const auto* import_stmt = dynamic_cast<const ast::ImportStatement*>(&stmt)) {
        visit_import_statement(*import_stmt);
    } else if (const auto* export_stmt = dynamic_cast<const ast::ExportStatement*>(&stmt)) {
        visit_export_statement(*export_stmt);
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

Value Interpreter::visit_boolean_literal(const ast::BooleanLiteral& node) {
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
    if (node.op == "&&") {
        bool left_bool = std::any_cast<bool>(left);
        bool right_bool = std::any_cast<bool>(right);
        return left_bool && right_bool;
    }
    if (node.op == "||") {
        bool left_bool = std::any_cast<bool>(left);
        bool right_bool = std::any_cast<bool>(right);
        return left_bool || right_bool;
    }
    
    throw std::runtime_error("Unknown binary operator: " + node.op);
}

Value Interpreter::visit_unary_op(const ast::UnaryOp& node) {
    Value right = evaluate(*node.right);
    
    if (node.op == "-") {
        double right_num = std::any_cast<double>(right);
        return -right_num;
    }
    if (node.op == "!") {
        bool right_bool = std::any_cast<bool>(right);
        return !right_bool;
    }
    
    throw std::runtime_error("Unknown unary operator: " + node.op);
}

Value Interpreter::visit_call(const ast::Call& node) {
    std::vector<Value> arguments;
    for (const auto& arg : node.args) {
        arguments.push_back(evaluate(*arg));
    }

    // First check if it's a user-defined function
    try {
        Value callee = environment_->get(node.callee);
        if (auto function = std::any_cast<std::shared_ptr<UserFunction>>(callee)) {
            return function->call(*this, arguments);
        }
    } catch (const std::runtime_error& e) {
        // Not a user-defined function, try builtin
    } catch (const std::bad_any_cast&) {
        // Not a user-defined function, try builtin
    }
    
    // Fall back to builtin function
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
        environment_->define(input, 0.0);
    }
    
    // If this pattern inherits from another pattern, find and execute it first
    if (!decl.parent_pattern.empty()) {
        // Find the parent pattern in the program
        bool found_parent = false;
        for (const auto& pattern : program_->patterns) {
            if (pattern->name == decl.parent_pattern) {
                // Execute parent pattern in its own environment
                Environment parent_env(&globals_);  // Parent pattern uses globals as enclosing scope
                Environment* previous = environment_;
                environment_ = &parent_env;
                
                // Execute parent pattern
                execute_pattern_decl(*pattern);
                
                // Restore environment
                environment_ = previous;
                
                try {
                    // Get parent pattern result from globals
                    Value parent_result = globals_.get(decl.parent_pattern);
                    PatternResult parent_pattern = std::any_cast<PatternResult>(parent_result);
                    
                    // Copy parent pattern variables into current environment
                    for (const auto& [name, value] : parent_pattern) {
                        environment_->define(name, value);
                    }
                } catch (const std::exception& e) {
                    throw std::runtime_error("Failed to inherit from pattern '" + decl.parent_pattern + "': " + e.what());
                }
                
                found_parent = true;
                break;
            }
        }
        
        if (!found_parent) {
            throw std::runtime_error("Parent pattern '" + decl.parent_pattern + "' not found");
        }
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
    // First check the dynamic built-ins map
    auto it = builtins_.find(name);
    if (it != builtins_.end()) {
        return it->second(args);
    }
    
    // Fall back to legacy hardcoded functions (TODO: migrate all to builtins_ map)
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
            throw std::runtime_error("Engine call failed for measure_coherence");
        }
    }
    
    if (name == "qfh_analyze") {
        if (args.empty()) {
            throw std::runtime_error("qfh_analyze expects a bitstream argument");
        }

        // Convert the DSL bitstream argument to a std::vector<uint8_t>
        std::vector<uint8_t> bitstream;
        try {
            std::string bitstream_str = std::any_cast<std::string>(args[0]);
            for (char c : bitstream_str) {
                bitstream.push_back(c - '0');
            }
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid bitstream argument for qfh_analyze");
        }

        // Call the engine facade
        sep::engine::QFHAnalysisRequest request;
        request.bitstream = bitstream;
        sep::engine::QFHAnalysisResponse response;
        auto result = engine.qfhAnalyze(request, response);

        if (sep::core::isSuccess(result)) {
            // For now, we'll return the rupture ratio as the primary result
            return static_cast<double>(response.rupture_ratio);
        } else {
            throw std::runtime_error("Engine call failed for qfh_analyze");
        }
    }
    
    if (name == "measure_entropy") {
        std::cout << "Calling real measure_entropy with " << args.size() << " arguments" << std::endl;
        
        // Convert DSL arguments into the request struct
        sep::engine::PatternAnalysisRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "entropy_pattern";
            }
        }
        request.analysis_depth = 2;
        request.include_relationships = false;
        
        // Call the real C++ function to get pattern metrics
        sep::engine::PatternAnalysisResponse response;
        auto result = engine.analyzePattern(request, response);
        
        if (sep::core::isSuccess(result)) {
            // Return real entropy from QFH analysis
            std::cout << "Real entropy from engine: " << response.entropy << std::endl;
            return static_cast<double>(response.entropy);
        } else {
            throw std::runtime_error("Engine call failed for measure_entropy");
        }
    }
    
    if (name == "extract_bits") {
        std::cout << "Calling real extract_bits with " << args.size() << " arguments" << std::endl;
        
        // Convert DSL arguments into the bit extraction request
        sep::engine::BitExtractionRequest request;
        if (!args.empty()) {
            try {
                request.pattern_id = std::any_cast<std::string>(args[0]);
            } catch (const std::bad_any_cast&) {
                request.pattern_id = "bitstream_pattern";
            }
        }
        
        // Call the real bit extraction engine
        sep::engine::BitExtractionResponse response;
        auto result = engine.extractBits(request, response);
        
        if (sep::core::isSuccess(result) && response.success) {
            // Convert bitstream to string representation for DSL
            std::string bitstream;
            for (uint8_t bit : response.bitstream) {
                bitstream += (bit == 1) ? '1' : '0';
            }
            
            std::cout << "Real bit extraction - extracted " << response.bitstream.size() << " bits" << std::endl;
            return bitstream;
        } else {
            throw std::runtime_error("Engine call failed for extract_bits: " + response.error_message);
        }
    }

    if (name == "manifold_optimize") {
        if (args.empty()) {
            throw std::runtime_error("manifold_optimize expects a pattern_id argument");
        }

        // Get the pattern_id from the DSL arguments
        std::string pattern_id;
        try {
            pattern_id = std::any_cast<std::string>(args[0]);
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Invalid pattern_id argument for manifold_optimize");
        }

        // Call the engine facade
        sep::engine::ManifoldOptimizationRequest request;
        request.pattern_id = pattern_id;
        sep::engine::ManifoldOptimizationResponse response;
        auto result = engine.manifoldOptimize(request, response);

        if (sep::core::isSuccess(result)) {
            // For now, we'll return a boolean indicating success
            return response.success;
        } else {
            throw std::runtime_error("Engine call failed for manifold_optimize");
        }
    }
    
    // ============================================================================
    // Type Checking & Conversion Functions (TASK.md Phase 2A Priority 1)
    // ============================================================================
    if (name == "is_number") {
        if (args.empty()) {
            throw std::runtime_error("is_number() requires exactly 1 argument");
        }
        try {
            std::any_cast<double>(args[0]);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    
    if (name == "is_string") {
        if (args.empty()) {
            throw std::runtime_error("is_string() requires exactly 1 argument");
        }
        try {
            std::any_cast<std::string>(args[0]);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    
    if (name == "is_bool") {
        if (args.empty()) {
            throw std::runtime_error("is_bool() requires exactly 1 argument");
        }
        try {
            std::any_cast<bool>(args[0]);
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    
    if (name == "to_string") {
        if (args.empty()) {
            throw std::runtime_error("to_string() requires exactly 1 argument");
        }
        return stringify(args[0]);
    }
    
    if (name == "to_number") {
        if (args.empty()) {
            throw std::runtime_error("to_number() requires exactly 1 argument");
        }
        try {
            return std::any_cast<double>(args[0]);
        } catch (const std::bad_any_cast&) {
            try {
                std::string str = std::any_cast<std::string>(args[0]);
                return std::stod(str);
            } catch (const std::exception&) {
                try {
                    bool b = std::any_cast<bool>(args[0]);
                    return b ? 1.0 : 0.0;
                } catch (const std::bad_any_cast&) {
                    throw std::runtime_error("Cannot convert this type to number");
                }
            }
        }
    }
    
    // ============================================================================
    // Math Functions
    // ============================================================================
    if (name == "abs") {
        if (args.size() != 1) {
            throw std::runtime_error("abs() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::abs(x);
    }
    
    if (name == "min") {
        if (args.size() != 2) {
            throw std::runtime_error("min() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::min(a, b);
    }
    
    if (name == "max") {
        if (args.size() != 2) {
            throw std::runtime_error("max() expects exactly 2 arguments");
        }
        double a = std::any_cast<double>(args[0]);
        double b = std::any_cast<double>(args[1]);
        return std::max(a, b);
    }
    
    if (name == "sqrt") {
        if (args.size() != 1) {
            throw std::runtime_error("sqrt() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        if (x < 0.0) {
            throw std::runtime_error("sqrt() domain error: argument must be non-negative");
        }
        return std::sqrt(x);
    }
    
    if (name == "pow") {
        if (args.size() != 2) {
            throw std::runtime_error("pow() expects exactly 2 arguments");
        }
        double x = std::any_cast<double>(args[0]);
        double y = std::any_cast<double>(args[1]);
        return std::pow(x, y);
    }
    
    if (name == "sin") {
        if (args.size() != 1) {
            throw std::runtime_error("sin() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::sin(x);
    }
    
    if (name == "cos") {
        if (args.size() != 1) {
            throw std::runtime_error("cos() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::cos(x);
    }
    
    if (name == "round") {
        if (args.size() != 1) {
            throw std::runtime_error("round() expects exactly 1 argument");
        }
        double x = std::any_cast<double>(args[0]);
        return std::round(x);
    }
    
    // ============================================================================
    // Statistical Functions
    // ============================================================================
    if (name == "mean") {
        if (args.empty()) {
            throw std::runtime_error("mean() requires at least 1 argument");
        }
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        return sum / args.size();
    }
    
    if (name == "median") {
        if (args.empty()) {
            throw std::runtime_error("median() requires at least 1 argument");
        }
        std::vector<double> values;
        for (const auto& arg : args) {
            values.push_back(std::any_cast<double>(arg));
        }
        std::sort(values.begin(), values.end());
        
        size_t n = values.size();
        if (n % 2 == 0) {
            return (values[n/2 - 1] + values[n/2]) / 2.0;
        } else {
            return values[n/2];
        }
    }
    
    if (name == "std_dev") {
        if (args.size() < 2) {
            throw std::runtime_error("std_dev() requires at least 2 arguments");
        }
        
        // Calculate mean
        double sum = 0.0;
        for (const auto& arg : args) {
            sum += std::any_cast<double>(arg);
        }
        double mean = sum / args.size();
        
        // Calculate sum of squared differences
        double sum_sq_diff = 0.0;
        for (const auto& arg : args) {
            double x = std::any_cast<double>(arg);
            double diff = x - mean;
            sum_sq_diff += diff * diff;
        }
        
        // Use sample standard deviation (divide by n-1)
        return std::sqrt(sum_sq_diff / (args.size() - 1));
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

Value Interpreter::visit_weighted_sum(const ast::WeightedSum& node) {
   double total = 0.0;
   for (const auto& pair : node.pairs) {
       Value weight_val = evaluate(*pair.first);
       Value value_val = evaluate(*pair.second);
       
       double weight = std::any_cast<double>(weight_val);
       double value = std::any_cast<double>(value_val);
       
       total += weight * value;
   }
   return total;
}

void Interpreter::visit_evolve_statement(const ast::EvolveStatement& node) {
    Value condition_result = evaluate(*node.condition);
    
    if (is_truthy(condition_result)) {
        // Create a new environment for the evolve block
        Environment evolve_env(environment_);
        Environment* previous = environment_;
        environment_ = &evolve_env;
        
        // Execute the evolve block
        for (const auto& stmt : node.body) {
            execute(*stmt);
        }
        
        // Restore previous environment
        environment_ = previous;
    }
}

void Interpreter::visit_if_statement(const ast::IfStatement& node) {
    Value condition_result = evaluate(*node.condition);
    
    if (is_truthy(condition_result)) {
        // Create a new environment for the then branch
        Environment then_env(environment_);
        Environment* previous = environment_;
        environment_ = &then_env;
        
        // Execute the then branch
        for (const auto& stmt : node.then_branch) {
            execute(*stmt);
        }
        
        // Restore previous environment
        environment_ = previous;
    } else if (!node.else_branch.empty()) {
        // Create a new environment for the else branch
        Environment else_env(environment_);
        Environment* previous = environment_;
        environment_ = &else_env;
        
        // Execute the else branch
        for (const auto& stmt : node.else_branch) {
            execute(*stmt);
        }
        
        // Restore previous environment
        environment_ = previous;
    }
}

void Interpreter::visit_while_statement(const ast::WhileStatement& node) {
    // Create a new environment for the while block
    Environment while_env(environment_);
    Environment* previous = environment_;
    environment_ = &while_env;
    
    // Execute the while loop
    while (is_truthy(evaluate(*node.condition))) {
        for (const auto& stmt : node.body) {
            execute(*stmt);
        }
    }
    
    // Restore previous environment
    environment_ = previous;
}

Value UserFunction::call(Interpreter& interpreter, const std::vector<Value>& arguments) {
    // Create a new environment for the function execution
    Environment function_env(closure_);
    Environment* previous = interpreter.environment_;
    interpreter.environment_ = &function_env;

    // Bind arguments to parameters
    for (size_t i = 0; i < declaration_.parameters.size(); i++) {
        if (i < arguments.size()) {
            function_env.define(declaration_.parameters[i], arguments[i]);
        } else {
            function_env.define(declaration_.parameters[i], nullptr); // Default value for missing args
        }
    }

    try {
        // Execute function body
        for (const auto& stmt : declaration_.body) {
            interpreter.execute(*stmt);
        }
        // If no return statement was encountered, return null
        interpreter.environment_ = previous;
        return nullptr;
    } catch (const ReturnException& return_value) {
        // Handle return statement
        interpreter.environment_ = previous;
        return return_value.value();
    }
}

void Interpreter::visit_function_declaration(const ast::FunctionDeclaration& node) {
    auto function = std::make_shared<UserFunction>(node, environment_);
    environment_->define(node.name, function);
}

void Interpreter::visit_return_statement(const ast::ReturnStatement& node) {
    Value value = node.value ? evaluate(*node.value) : nullptr;
    throw ReturnException(value);
}

void Interpreter::visit_import_statement(const ast::ImportStatement& node) {
    std::cout << "Importing module: " << node.module_path << std::endl;
    
    // For now, just implement a basic file-based import system
    // In a full implementation, this would parse and execute the imported file
    // and bring the specified exports into the current environment
    
    if (!node.imports.empty()) {
        std::cout << "Importing specific items: ";
        for (const auto& import : node.imports) {
            std::cout << import << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Importing all exports from module" << std::endl;
    }
    
    // TODO: Implement actual module loading and namespace management
    // This would involve:
    // 1. Parse the module file
    // 2. Execute it in an isolated environment
    // 3. Extract the exported patterns/functions
    // 4. Import them into the current environment
}

void Interpreter::visit_export_statement(const ast::ExportStatement& node) {
    std::cout << "Exporting: ";
    for (const auto& export_name : node.exports) {
        std::cout << export_name << " ";
    }
    std::cout << std::endl;
    
    // TODO: Implement actual export tracking
    // This would involve marking the specified variables/patterns/functions
    // as available for import by other modules
}

// Variable access methods
Value Interpreter::get_global_variable(const std::string& name) {
    return globals_.get(name);
}

bool Interpreter::has_global_variable(const std::string& name) {
    try {
        globals_.get(name);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

const std::unordered_map<std::string, Value>& Interpreter::get_global_variables() const {
    return globals_.getVariables();
}

} // namespace dsl::runtime
