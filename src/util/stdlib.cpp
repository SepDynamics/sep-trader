#include "util/stdlib.h"
#include <iostream>

namespace dsl::stdlib {

void register_all(Context& context) {
    std::cout << "Registering DSL standard library functions..." << std::endl;
    
    // Register implemented modules
    dsl::stdlib::register_core_primitives(context);
    
    // Note: sep_dsl functions expect VMExecutionContext, not dsl::compiler::Context
    // For now, we'll skip these until proper adapter is implemented
    // sep_dsl::stdlib::register_math(context);
    // sep_dsl::stdlib::register_statistical(context);
    
    // TODO: Implement these modules in future phases
    // register_time_series(context);
    // register_pattern_matching(context);
    // register_data_transformation(context);
    // register_aggregation(context);
    
    std::cout << "DSL standard library registration complete!" << std::endl;
}

}
