#pragma once

#include "util/compiler.h"

namespace dsl::stdlib {

using dsl::compiler::Context;

/// Register time series utilities with the compiler context
void register_time_series(Context& context);

} // namespace dsl::stdlib
