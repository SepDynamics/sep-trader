add_sep_test(historical_candles_test
    SOURCES historical_candles_test.cpp
    DEPENDENCIES sep_lib
)

add_sep_test(valkey_url_parse_test
    SOURCES valkey_url_parse_test.cpp
    DEPENDENCIES sep_lib
)
