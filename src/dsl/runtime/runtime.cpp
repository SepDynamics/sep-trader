#include "runtime.h"
#include <iostream>
#include <stdexcept>

namespace sep::dsl::runtime {

    DSLRuntime::DSLRuntime() 
        : parser_(std::make_unique<parser::SEPParser>())
        , compiler_(std::make_unique<compiler::SEPCompiler>())
        , engine_facade_(engine::EngineFacade::getInstance())
        , memory_manager_(memory::MemoryTierManager::getInstance())
        , debug_mode_(false) {
        
        logDebug("DSL Runtime initialized");
    }

    DSLRuntime::~DSLRuntime() {
        logDebug("DSL Runtime shutting down");
    }

    bool DSLRuntime::executeScript(const std::string& script_source) {
        runtime_errors_.clear();
        
        try {
            // Parse the script
            logDebug("Parsing DSL script...");
            auto program = parser_->parse(script_source);
            
            if (parser_->hasErrors()) {
                for (const auto& error : parser_->getErrors()) {
                    addError("Parser error: " + error);
                }
                return false;
            }
            
            // Compile to executable operations
            logDebug("Compiling DSL program...");
            auto compiled_program = compiler_->compile(*program);
            
            if (compiler_->hasErrors()) {
                for (const auto& error : compiler_->getErrors()) {
                    addError("Compiler error: " + error);
                }
                return false;
            }
            
            // Execute the compiled program
            logDebug("Executing compiled program...");
            return executeProgram(compiled_program);
            
        } catch (const std::exception& e) {
            addError("Runtime exception: " + std::string(e.what()));
            return false;
        }
    }

    bool DSLRuntime::executeProgram(const compiler::CompiledProgram& program) {
        bool success = true;
        
        for (const auto& operation : program) {
            if (!executeOperation(operation)) {
                success = false;
                // Continue executing other operations for now
                // TODO: Add option for fail-fast vs. continue-on-error
            }
        }
        
        return success;
    }

    bool DSLRuntime::executePattern(const ast::PatternNode& pattern) {
        try {
            // Compile just this pattern
            auto operation = compiler_->compilePattern(pattern);
            return executeOperation(operation);
            
        } catch (const std::exception& e) {
            addError("Pattern execution error: " + std::string(e.what()));
            return false;
        }
    }

    bool DSLRuntime::hasErrors() const {
        return !runtime_errors_.empty() || 
               parser_->hasErrors() || 
               compiler_->hasErrors();
    }

    std::vector<std::string> DSLRuntime::getErrors() const {
        std::vector<std::string> all_errors = runtime_errors_;
        
        // Collect parser errors
        for (const auto& error : parser_->getErrors()) {
            all_errors.push_back("Parser: " + error);
        }
        
        // Collect compiler errors
        for (const auto& error : compiler_->getErrors()) {
            all_errors.push_back("Compiler: " + error);
        }
        
        return all_errors;
    }

    void DSLRuntime::reset() {
        runtime_errors_.clear();
        // TODO: Reset parser and compiler state if needed
        logDebug("DSL Runtime reset");
    }

    void DSLRuntime::logDebug(const std::string& message) {
        if (debug_mode_) {
            std::cout << "[DSL Runtime] " << message << std::endl;
        }
    }

    void DSLRuntime::addError(const std::string& error) {
        runtime_errors_.push_back(error);
        if (debug_mode_) {
            std::cerr << "[DSL Runtime Error] " << error << std::endl;
        }
    }

    bool DSLRuntime::executeOperation(const compiler::EngineOperation& operation) {
        try {
            // Execute the operation against the engine facade
            operation(engine_facade_);
            return true;
            
        } catch (const std::exception& e) {
            addError("Operation execution failed: " + std::string(e.what()));
            return false;
        }
    }

    // Convenience functions implementation
    namespace convenience {
        
        bool executePatternString(const std::string& pattern_def) {
            DSLRuntime runtime;
            runtime.setDebugMode(true);
            
            // For now, treat the pattern as a complete program
            // TODO: Wrap single patterns in program structure
            return runtime.executeScript(pattern_def);
        }
        
        bool executeProgramString(const std::string& program) {
            DSLRuntime runtime;
            return runtime.executeScript(program);
        }
        
        bool validateScript(const std::string& script) {
            DSLRuntime runtime;
            parser::SEPParser parser;
            
            auto program = parser.parse(script);
            return !parser.hasErrors();
        }
    }

} // namespace sep::dsl::runtime
