#pragma once

#ifndef SRC_CORE_CANDLE_DATA_MERGED_H
#define SRC_CORE_CANDLE_DATA_MERGED_H

#ifndef SRC_CORE_CANDLE_DATA_H
#define SRC_CORE_CANDLE_DATA_H
#include <cstdint>
namespace sep {
    struct CandleData {
        uint64_t timestamp;
        double open;
        double high;
        double low;
        double close;
        double volume;
    };
}
#endif

#ifndef SRC_CORE_INTERNAL_CANDLE_DATA_H
#define SRC_CORE_INTERNAL_CANDLE_DATA_H
namespace sep {
    struct InternalCandleData : public CandleData {
        double some_internal_value;
    };
}
#endif

#ifndef SRC_UTIL_CANDLE_DATA_H
#define SRC_UTIL_CANDLE_DATA_H
#include <vector>
namespace sep {
    using CandleHistory = std::vector<CandleData>;
}
#endif

#endif // SRC_CORE_CANDLE_DATA_MERGED_H