#pragma once

#include "core_primitives.h"
#include "math/math.h"
#include "statistical/statistical.h"
#include "time_series/time_series.h"
#include "pattern_matching/pattern_matching.h"
#include "data_transformation/data_transformation.h"
#include "aggregation/aggregation.h"

namespace sep::dsl::stdlib {

void register_all(Runtime& runtime);

}
