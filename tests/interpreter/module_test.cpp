#include "gtest/gtest.h"
#include "util/interpreter.h"
#include <memory>
#include <vector>

using dsl::runtime::Interpreter;
using dsl::runtime::Value;
using dsl::runtime::PatternResult;

TEST(Interpreter, PatternInputsAndModuleLoading) {
    Interpreter interp;

    auto provider = std::make_unique<dsl::ast::PatternDecl>();
    provider->name = "provider";
    provider->inputs = {"a", "b"};

    auto func = std::make_unique<dsl::ast::FunctionDeclaration>();
    func->name = "foo";
    auto ret = std::make_unique<dsl::ast::ReturnStatement>();
    auto num = std::make_unique<dsl::ast::NumberLiteral>();
    num->value = 5.0;
    ret->value = std::move(num);
    func->body.push_back(std::move(ret));

    auto exp = std::make_unique<dsl::ast::ExportStatement>();
    exp->exports.push_back("foo");

    provider->body.push_back(std::move(func));
    provider->body.push_back(std::move(exp));

    auto consumer = std::make_unique<dsl::ast::PatternDecl>();
    consumer->name = "consumer";

    auto imp = std::make_unique<dsl::ast::ImportStatement>();
    imp->module_path = "";
    imp->imports = {"foo"};

    auto call = std::make_unique<dsl::ast::Call>();
    call->callee = "foo";
    auto assign = std::make_unique<dsl::ast::Assignment>();
    assign->name = "x";
    assign->value = std::move(call);

    consumer->body.push_back(std::move(imp));
    consumer->body.push_back(std::move(assign));

    dsl::ast::Program program;
    program.patterns.push_back(std::move(provider));
    program.patterns.push_back(std::move(consumer));

    interp.interpret(program);

    PatternResult providerRes = std::any_cast<PatternResult>(interp.get_global_variable("provider"));
    auto inputs = std::any_cast<std::vector<Value>>(providerRes.at("inputs"));
    EXPECT_EQ(inputs.size(), 2u);

    PatternResult consumerRes = std::any_cast<PatternResult>(interp.get_global_variable("consumer"));
    double x = std::any_cast<double>(consumerRes.at("x"));
    EXPECT_DOUBLE_EQ(x, 5.0);
}
