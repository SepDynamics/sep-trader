add_sep_library(sep_core_types
    SOURCES
        pattern_conversions.cpp
        result.cpp
    PCH_HEADER ../common/sep_precompiled.h
)

target_include_directories(sep_core_types PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${spdlog_SOURCE_DIR}/include
    ${fmt_SOURCE_DIR}/include
)