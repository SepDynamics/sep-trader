#include "stdlib.h"
#include <iostream>

namespace dsl::stdlib {

void register_all(Context& context) {
    std::cout << "Registering DSL standard library functions..." << std::endl;
    
    // Register implemented modules
    register_core_primitives(context);
    register_math(context);
    register_statistical(context);
    
    // TODO: Implement these modules in future phases
    // register_time_series(context);
    // register_pattern_matching(context);
    // register_data_transformation(context);
    // register_aggregation(context);
    
    std::cout << "DSL standard library registration complete!" << std::endl;
}

}
