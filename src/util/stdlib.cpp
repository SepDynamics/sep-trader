#include "util/stdlib.h"
#include "util/time_series.h"

namespace dsl::stdlib {

void register_all(Context& context) {
    // Register implemented modules
    dsl::stdlib::register_core_primitives(context);
    register_time_series(context);
}

} // namespace dsl::stdlib
