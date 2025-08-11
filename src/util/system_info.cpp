#include "system_info.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unistd.h>

namespace sep::util {

// Returns the total number of clock ticks since boot
double get_total_ticks() {
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        return 0.0;
    }

    std::string line;
    std::getline(file, line);
    file.close();

    std::string cpu_str;
    long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    sscanf(line.c_str(), "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", &cpu_str[0], &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal, &guest, &guest_nice);

    return static_cast<double>(user + nice + system + idle + iowait + irq + softirq + steal);
}

// Returns the number of idle clock ticks since boot
double get_idle_ticks() {
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        return 0.0;
    }

    std::string line;
    std::getline(file, line);
    file.close();

    std::string cpu_str;
    long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
    sscanf(line.c_str(), "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", &cpu_str[0], &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal, &guest, &guest_nice);

    return static_cast<double>(idle + iowait);
}

double get_cpu_usage() {
    double total_ticks1 = get_total_ticks();
    double idle_ticks1 = get_idle_ticks();

    usleep(1000000); // Sleep for 1 second

    double total_ticks2 = get_total_ticks();
    double idle_ticks2 = get_idle_ticks();

    double total_delta = total_ticks2 - total_ticks1;
    double idle_delta = idle_ticks2 - idle_ticks1;

    if (total_delta == 0.0) {
        return 0.0;
    }

    return 100.0 * (1.0 - (idle_delta / total_delta));
}

} // namespace sep::util
