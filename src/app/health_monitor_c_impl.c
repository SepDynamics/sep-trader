/* Pure C implementation of health monitoring to bypass type system pollution */
#include "health_monitor_c_wrapper.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Mock implementation since we can't safely include C++ headers */
static int initialized = 0;

int c_health_monitor_init() {
    initialized = 1;
    return 1;
}

int c_health_monitor_get_status(CHealthStatus* status) {
    if (!initialized || !status) {
        return 0;
    }
    
    /* Provide mock health status - in real implementation this would call C++ */
    status->is_healthy = 1;
    status->memory_usage_mb = 64.5;
    status->hit_rate = 0.85;
    status->cache_size = 12345;
    strcpy(status->issues, "System operational - mock data");
    
    return 1;
}

void c_health_monitor_cleanup() {
    initialized = 0;
}