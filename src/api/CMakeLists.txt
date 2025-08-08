# SEP API Component - State Controller and Management API
# This component provides RESTful API interface for professional trading control

project(sep_api)

# API library
add_library(sep_api STATIC
    state_controller.cpp
)

target_include_directories(sep_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(sep_api PUBLIC
    sep_core
    sep_config
    Threads::Threads
    sep_global_includes
    TBB::tbb
)

# Enable C++17 features
target_compile_features(sep_api PUBLIC cxx_std_17)

# Set compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(sep_api PRIVATE 
        -Wall 
        -Wextra 
        -Wpedantic
        -O3
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(sep_api PRIVATE 
        /W4
        /O2
    )
endif()

# Fixed by proper include order in standard_includes.h

# Install targets
install(TARGETS sep_api
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    state_controller.hpp
    DESTINATION include/sep/api
)
