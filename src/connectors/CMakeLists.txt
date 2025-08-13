add_sep_library(sep_connectors
    SOURCES
        oanda_connector.cpp
        market_data_converter.cpp
        qdrant_connector.cpp
    PCH_HEADER ../common/stable_headers.h
    DEPENDENCIES
        CURL::libcurl
        TBB::tbb
        sep_engine
        nlohmann_json::nlohmann_json
)

target_include_directories(sep_connectors PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${spdlog_SOURCE_DIR}/include
    /usr/include/c++/11
)

# Build the Qdrant test executable
add_executable(test_qdrant test_qdrant.cpp)
target_link_libraries(test_qdrant PRIVATE
    sep_connectors
    sep_core_types
    CURL::libcurl
    nlohmann_json::nlohmann_json
)