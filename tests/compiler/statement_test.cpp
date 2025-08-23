#include "gtest/gtest.h"
#include "util/compiler.h"
#include "util/nodes.h"
#include "util/time_series.h"
#include <memory>
#include <vector>

using dsl::compiler::Compiler;
using dsl::compiler::Context;
using dsl::compiler::Value;

TEST(Compiler, AssignmentAndExpression) {
    Compiler compiler;
    Context ctx;
    dsl::stdlib::register_time_series(ctx);

    auto setVar = [&ctx](const std::vector<Value>& args) -> Value {
        if (!args.empty()) {
            ctx.set_variable("y", args[0]);
        }
        return Value{};
    };
    ctx.register_function("setVar", setVar);

    auto num = std::make_unique<dsl::ast::NumberLiteral>();
    num->value = 42.0;
    dsl::ast::Assignment assign;
    assign.name = "x";
    assign.value = std::move(num);
    auto assignFn = compiler.compile_statement(assign);
    assignFn(ctx);
    Value x = ctx.get_variable("x");
    ASSERT_TRUE(std::holds_alternative<double>(x));
    EXPECT_DOUBLE_EQ(std::get<double>(x), 42.0);

    auto arg = std::make_unique<dsl::ast::NumberLiteral>();
    arg->value = 7.0;
    auto call = std::make_unique<dsl::ast::Call>();
    call->callee = "setVar";
    call->args.push_back(std::move(arg));
    dsl::ast::ExpressionStatement exprStmt;
    exprStmt.expression = std::move(call);
    auto exprFn = compiler.compile_statement(exprStmt);
    exprFn(ctx);
    Value y = ctx.get_variable("y");
    ASSERT_TRUE(std::holds_alternative<double>(y));
    EXPECT_DOUBLE_EQ(std::get<double>(y), 7.0);
}
