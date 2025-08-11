# SEP Apps - Standalone Applications

# ENABLED: data_downloader
add_executable(data_downloader data_downloader.cpp)
set_target_properties(data_downloader PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

# REMOVED: Problematic forced include

target_include_directories(data_downloader PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(data_downloader PRIVATE
    # sep_trading  # Temporarily disabled
    sep_connectors
    sep_common
    sep_core_integrated
    sep_quantum
    sep_engine
    sep_core_deps
)

# ENABLED: oanda_trader subdirectory
add_subdirectory(oanda_trader)
