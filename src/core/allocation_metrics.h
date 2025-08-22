#pragma once

#include "core/standard_includes.h"
#include "core/prometheus_exporter.h"

namespace sep::metrics {

/// Global counter tracking memory allocation failures.
Counter& allocationFailures();

}  // namespace sep::metrics
