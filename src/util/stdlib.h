#pragma once

#include "util/core_primitives.h"
#include "math.h"
#include "statistical.h"
#include "vm.h"

namespace sep_dsl::stdlib {

// Import Context type
using Context = sep_dsl::bytecode::VMExecutionContext;

/// Register all standard library functions with the runtime context
void register_all(Context& context);

}
