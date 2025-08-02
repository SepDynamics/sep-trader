#pragma once

#include "dsl/ast/nodes.h"
#include <functional>
#include <vector>
#include <memory>
#include <unordered_set>

// Forward declarations for the existing SEP engine
namespace sep {
    namespace engine { class EngineFacade; }
    namespace memory { class MemoryTierManager; }
    namespace quantum { class QuantumProcessor; }
}

namespace sep::dsl::compiler {

    // A compiled program is a series of operations to execute against the engine
    using EngineOperation = std::function<void(engine::EngineFacade&)>;
    using CompiledProgram = std::vector<EngineOperation>;

    class SEPCompiler {
    public:
        SEPCompiler();
        ~SEPCompiler();

        // Main compilation entry point
        CompiledProgram compile(const ast::Program& program);
        
        // Individual compilation methods for development/testing
        EngineOperation compilePattern(const ast::PatternNode& pattern);
        EngineOperation compileSignal(const ast::SignalNode& signal);
        EngineOperation compileMemoryRule(const ast::MemoryRule& rule);
        
        // Error handling
        bool hasErrors() const;
        std::vector<std::string> getErrors() const;

    private:
        std::vector<std::string> errors_;
        
        // Internal compilation helpers
        std::string generateVariableName(const std::string& base);
        
        // Expression compilation
        std::string compileExpression(const ast::Expression& expr);
        std::string compileLiteral(const ast::LiteralExpression& literal);
        std::string compileVariable(const ast::VariableExpression& var);
        std::string compileBinaryOp(const ast::BinaryExpression& binary);
        std::string compileFunctionCall(const ast::FunctionCallExpression& call);
        
        // Operation compilation - these map to your existing C++ primitives
        std::string compileQFHAnalyze(const std::vector<std::string>& args);
        std::string compileQBSAAnalyze(const std::vector<std::string>& args);
        std::string compileManifoldOptimize(const std::vector<std::string>& args);
        std::string compileMeasureCoherence(const std::vector<std::string>& args);
        std::string compileMeasureStability(const std::vector<std::string>& args);
        std::string compileMeasureEntropy(const std::vector<std::string>& args);
        
        // Memory operation compilation
        std::string compileMemoryStore(const std::string& pattern_var, 
                                     const std::string& tier);
        std::string compileMemoryRetrieve(const std::string& id_var);
        
        // Variable tracking for scoping
        std::unordered_set<std::string> declared_variables_;
        size_t temp_var_counter_;
    };

} // namespace sep::dsl::compiler
