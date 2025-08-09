# SEP Apps - Standalone Applications

# ENABLED: data_downloader
add_executable(data_downloader data_downloader.cpp)

# REMOVED: Problematic forced include

target_link_libraries(data_downloader PRIVATE
    sep_trading
    sep_connectors
    sep_common
    sep_core
    sep_config
    sep_cache
    sep_quantum
    sep_engine
    sep_core_deps
)

# ENABLED: oanda_trader subdirectory
add_subdirectory(oanda_trader)
