# SEP Apps - Standalone Applications

# ENABLED: data_downloader
add_executable(data_downloader data_downloader.cpp)
target_precompile_headers(data_downloader PRIVATE ../common/stable_headers.h)


# REMOVED: Problematic forced include

target_include_directories(data_downloader PRIVATE ${CMAKE_SOURCE_DIR}/src ${spdlog_SOURCE_DIR}/include /usr/include/c++/11)

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
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/oanda_trader/CMakeLists.txt)
    add_subdirectory(oanda_trader)
endif()
