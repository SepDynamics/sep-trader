#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// C-style health monitoring interface to bypass type pollution
typedef struct {
    int is_healthy;
    double memory_usage_mb;
    double hit_rate;
    long cache_size;
    char issues[256];
} CHealthStatus;

// C interface functions
int c_health_monitor_init();
int c_health_monitor_get_status(CHealthStatus* status);
void c_health_monitor_cleanup();

#ifdef __cplusplus
}
#endif