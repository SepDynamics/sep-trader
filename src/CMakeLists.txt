# Enforce header-only spdlog for all targets
add_definitions(-DSPDLOG_HEADER_ONLY)

# Add component subdirectories - core components needed for quant processing
add_subdirectory(core_types)
add_subdirectory(common)

# Standalone API server for headless engine operation (temporarily disabled)
# add_executable(sep_api_server
#     api_main.cpp
# )
# 
# target_include_directories(sep_api_server PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${CMAKE_SOURCE_DIR}/src
# )
# 
# target_link_libraries(sep_api_server PRIVATE
#     sep_api
#     sep_engine
#     sep_quantum
#     sep_memory
#     Threads::Threads
# )



