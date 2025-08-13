# Enforce header-only spdlog for all targets


# Add component subdirectories - core components needed for quant processing
add_subdirectory(core_types)
add_subdirectory(common)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core_integrated/CMakeLists.txt)
    add_subdirectory(core_integrated)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/metrics_library/CMakeLists.txt)
    add_subdirectory(metrics_library)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ui/CMakeLists.txt)
    add_subdirectory(ui)
endif()

# Core libraries always built

# Optional modules that require full environment (CUDA, external deps)
add_subdirectory(memory)
add_subdirectory(quantum)
add_subdirectory(engine)
if(NOT SEP_CPU_ONLY AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/CMakeLists.txt)
    add_subdirectory(cuda)
endif()
add_subdirectory(connectors)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/services/CMakeLists.txt)
    add_subdirectory(services)
endif()
add_compile_definitions(FMT_HEADER_ONLY)
# Temporarily disable trading module due to persistent std::array/functional issues
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/trading/CMakeLists.txt)
    add_subdirectory(trading)
endif()
add_subdirectory(dsl)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli/CMakeLists.txt)
    add_subdirectory(cli)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/CMakeLists.txt)
    add_subdirectory(apps)
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



