#pragma once

namespace sep::common
{

    struct CandleData
    {
        long long timestamp;  // UNIX timestamp
        double open;
        double high;
        double low;
        double close;
        long long volume;
        bool is_mock = false;  // Runtime flag to identify mock data
    };

}  // namespace sep::common