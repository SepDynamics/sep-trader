add_sep_library(sep_common
    SOURCES
        financial_data_types.cpp
    PCH_HEADER stable_headers.h
    DEPENDENCIES
        spdlog::spdlog
        fmt::fmt
)
target_include_directories(sep_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    /usr/include/c++/11
)

# Fixed by proper include order in standard_includes.h
