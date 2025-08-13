add_sep_library(sep_memory
    SOURCES
        memory_tier_manager_serialization.cpp
        memory_tier.cpp
        memory_tier_manager.cpp
        quantum_coherence_manager.cpp
        redis_manager.cpp
    DEPENDENCIES
        sep_core_types
        sep_common
        fmt::fmt
        Threads::Threads
        TBB::tbb
        CUDA::cudart
        CUDA::cuda_driver
        spdlog::spdlog
        hiredis
    PCH_HEADER ../common/stable_headers.h
)

target_include_directories(sep_memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${spdlog_SOURCE_DIR}/include
    /usr/include/c++/11
)

# Fixed by proper include order in standard_includes.h