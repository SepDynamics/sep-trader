#include "core/engine.h"

extern "C" double sep_get_total_patterns_processed() {
    static sep::core::Engine engine;
    auto metrics = engine.getMetrics();
    auto it = metrics.find("patterns_processed");
    if (it != metrics.end()) {
        return it->second;
    }
    return 0.0;
}
