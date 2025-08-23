#include "time_series.h"

#include <numeric>
#include <stdexcept>
#include <vector>

namespace dsl::stdlib {

using dsl::compiler::Value;

void register_time_series(Context& context) {
    context.register_function(
        "moving_average",
        [](const std::vector<Value>& args) -> Value {
            if (args.empty()) {
                return Value{};
            }
            double sum = 0.0;
            for (const auto& v : args) {
                if (!std::holds_alternative<double>(v)) {
                    throw std::runtime_error("moving_average expects numeric arguments");
                }
                sum += std::get<double>(v);
            }
            return Value(sum / static_cast<double>(args.size()));
        });
}

} // namespace dsl::stdlib
