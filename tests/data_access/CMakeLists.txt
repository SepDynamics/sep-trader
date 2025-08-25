add_sep_test(filter_test
    SOURCES filter_test.cpp
    DEPENDENCIES sep_lib
)

add_sep_test(historical_candles_test
    SOURCES historical_candles_test.cpp
    DEPENDENCIES sep_lib
)
