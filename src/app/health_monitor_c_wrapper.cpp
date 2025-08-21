// C-style wrapper implementation around CacheHealthMonitor
#include "health_monitor_c_wrapper.h"
#include <cstring>
#include "core/cache_health_monitor.hpp"

extern "C" {

static sep::cache::CacheHealthMonitor g_monitor;

int c_health_monitor_init() {
    return 1;  // Success
}

int c_health_monitor_get_status(CHealthStatus* status) {
    if (!status) {
        return 0;  // Failure - null pointer
    }

    auto health = g_monitor.getHealthStatus();
    status->is_healthy = health.is_healthy ? 1 : 0;
    status->memory_usage_mb = health.memory_usage_mb;
    status->hit_rate = health.hit_rate;
    status->cache_size = static_cast<long>(health.cache_size);
    strncpy(status->issues, health.issues.c_str(), 255);
    status->issues[255] = '\0';

    return 1;  // Success
}

void c_health_monitor_cleanup() {}

}  // extern "C"
