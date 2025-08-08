# Enforce header-only spdlog for all targets
add_definitions(-DSPDLOG_HEADER_ONLY)

# Add component subdirectories - core components needed for quant processing
add_subdirectory(core_types)
add_subdirectory(common)

# Only add subdirectories if their CMakeLists exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt)
    add_subdirectory(core)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config/CMakeLists.txt)
    add_subdirectory(config)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cache/CMakeLists.txt)
    add_subdirectory(cache)
endif()

# Core libraries always built

# Optional modules that require full environment (CUDA, external deps)
if(SEP_USE_CUDA)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/memory/CMakeLists.txt)
        add_subdirectory(memory)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/quantum/CMakeLists.txt)
        add_subdirectory(quantum)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/engine/CMakeLists.txt)
        add_subdirectory(engine)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/connectors/CMakeLists.txt)
        add_subdirectory(connectors)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trading/CMakeLists.txt)
        add_subdirectory(trading)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dsl/CMakeLists.txt)
        add_subdirectory(dsl)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli/CMakeLists.txt)
        add_subdirectory(cli)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/CMakeLists.txt)
        add_subdirectory(apps)
    endif()
else()
    message(STATUS "CUDA disabled; skipping optional modules")
endif()

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



