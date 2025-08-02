#pragma once

#include "dsl/ast/nodes.h"
#include "dsl/parser/parser.h"
#include "dsl/compiler/compiler.h"
#include "engine/facade/facade.h"
#include "memory/memory_tier_manager.hpp"
#include <memory>

namespace sep::dsl::runtime {

    class DSLRuntime {
    public:
        DSLRuntime();
        ~DSLRuntime();

        // High-level entry point for executing DSL scripts
        bool executeScript(const std::string& script_source);
        
        // Lower-level entry points for development/testing
        bool executeProgram(const compiler::CompiledProgram& program);
        bool executePattern(const ast::PatternNode& pattern);
        
        // DSL component access for advanced usage
        parser::SEPParser& getParser() { return *parser_; }
        compiler::SEPCompiler& getCompiler() { return *compiler_; }
        
        // Error handling
        bool hasErrors() const;
        std::vector<std::string> getErrors() const;
        
        // Runtime state management
        void reset();
        void setDebugMode(bool debug) { debug_mode_ = debug; }

    private:
        // DSL pipeline components
        std::unique_ptr<parser::SEPParser> parser_;
        std::unique_ptr<compiler::SEPCompiler> compiler_;
        
        // References to existing SEP engine singletons
        engine::EngineFacade& engine_facade_;
        memory::MemoryTierManager& memory_manager_;
        
        // Runtime state
        std::vector<std::string> runtime_errors_;
        bool debug_mode_;
        
        // Internal helpers
        void logDebug(const std::string& message);
        void addError(const std::string& error);
        
        // Execute individual operations safely
        bool executeOperation(const compiler::EngineOperation& operation);
    };

    // Convenience functions for quick DSL usage
    namespace convenience {
        // Execute a single pattern definition
        bool executePatternString(const std::string& pattern_def);
        
        // Execute a complete DSL program
        bool executeProgramString(const std::string& program);
        
        // Parse and validate DSL without execution (for development)
        bool validateScript(const std::string& script);
    }

} // namespace sep::dsl::runtime
