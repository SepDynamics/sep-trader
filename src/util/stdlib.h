#pragma once

#include "core_primitives.h"
#include "math.h"
#include "statistical.h"

namespace dsl::stdlib {

// Import Context type
using dsl::compiler::Context;

/// Register all standard library functions with the runtime context
void register_all(Context& context);

}
