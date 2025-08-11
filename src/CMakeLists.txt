# Enforce header-only spdlog for all targets


# Add component subdirectories - core components needed for quant processing
add_subdirectory(core_types)
add_subdirectory(common)

add_subdirectory(core_integrated)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ui/CMakeLists.txt)
    add_subdirectory(ui)
endif()

# Core libraries always built

# Optional modules that require full environment (CUDA, external deps)
add_subdirectory(memory)
add_subdirectory(quantum)
add_subdirectory(engine)
add_subdirectory(cuda)
add_subdirectory(connectors)
add_compile_definitions(FMT_HEADER_ONLY)
# Temporarily disable trading module due to persistent std::array/functional issues
# add_subdirectory(trading)
add_subdirectory(dsl)
add_subdirectory(cli)
add_subdirectory(apps)

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



