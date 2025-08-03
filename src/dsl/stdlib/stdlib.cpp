#include "stdlib.h"

namespace sep::dsl::stdlib {

void register_all(Runtime& runtime) {
    register_core_primitives(runtime);
    register_math(runtime);
    register_statistical(runtime);
    register_time_series(runtime);
    register_pattern_matching(runtime);
    register_data_transformation(runtime);
    register_aggregation(runtime);
}

}
