# SEP Apps - Standalone Applications

# ENABLED: data_downloader
add_executable(data_downloader data_downloader.cpp)
target_link_libraries(data_downloader PRIVATE
    sep_global_includes
    sep_trading
    sep_connectors
    sep_common
    sep_core
    sep_config
    sep_cache
    sep_quantum
    sep_engine
    nlohmann_json::nlohmann_json
    pqxx
    pq
    curl
    TBB::tbb
    spdlog::spdlog
    fmt::fmt
    hiredis
    Threads::Threads
)

# ENABLED: oanda_trader subdirectory
add_subdirectory(oanda_trader)
