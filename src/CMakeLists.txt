cmake_minimum_required(VERSION 3.18)
project(sep_trader LANGUAGES CXX C)

add_definitions(-DGLM_COMPILER=0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA)
if(CUDA_FOUND)
    set(USE_CUDA TRUE)
else()
    set(USE_CUDA FALSE)
    message(WARNING "CUDA not found, building without CUDA")
endif()

# Gather C++ source files only
file(GLOB_RECURSE CXX_SOURCES
    "app/*.cpp"
    "core/*.cpp"
    "cuda/*.cpp"
    "io/*.cpp"
    "util/*.cpp"
)

# Gather C source files separately
set(C_SOURCES
    "util/error_handling.c"
)

# Add CUDA sources if available
if(USE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES
        "app/*.cu" "core/*.cu" "cuda/*.cu" "io/*.cu" "util/*.cu")
    list(APPEND CXX_SOURCES ${CUDA_SOURCES})
endif()

# Exclude facade_original.cpp and OANDA-related files from the list of sources
list(FILTER CXX_SOURCES EXCLUDE REGEX ".*/facade_original\\.cpp$")
list(FILTER CXX_SOURCES EXCLUDE REGEX ".*/oanda_.*\\.(cpp|cu)$")

# Separate main files from library source files
set(LIB_CXX_SOURCES)
foreach(file ${CXX_SOURCES})
    if(NOT file MATCHES "main.cpp$")
        list(APPEND LIB_CXX_SOURCES ${file})
    endif()
endforeach()

# Create C library first (no PCH for C files)
add_library(sep_c_lib STATIC ${C_SOURCES})
set_source_files_properties(${C_SOURCES} PROPERTIES LANGUAGE C)
target_include_directories(sep_c_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
)

# Create C++ library with PCH
add_sep_library(sep_lib SOURCES ${LIB_CXX_SOURCES} PCH_HEADER core/sep_precompiled.h)
# Link C library to C++ library
target_link_libraries(sep_lib PRIVATE sep_c_lib)
target_link_libraries(sep_lib
    PUBLIC
        sep_core_deps
        $<$<BOOL:${USE_CUDA}>:CUDA::cudart>
    PRIVATE
        sep_fetchcontent_deps
)
# Note: Include directories are handled by add_sep_library template with proper generator expressions
# Note: sep_fetchcontent_deps is PRIVATE - available for building sep_lib but not part of export interface

# Collect public headers for installation
file(GLOB_RECURSE SEP_LIB_PUBLIC_HEADERS
    "app/*.h" "app/*.hpp" "app/*.cuh"
    "core/*.h" "core/*.hpp" "core/*.cuh"
    "cuda/*.h" "cuda/*.hpp" "cuda/*.cuh"
    "io/*.h" "io/*.hpp" "io/*.cuh"
    "util/*.h" "util/*.hpp" "util/*.cuh"
)
set_target_properties(sep_lib PROPERTIES PUBLIC_HEADER "${SEP_LIB_PUBLIC_HEADERS}")

install(TARGETS sep_lib
    PUBLIC_HEADER DESTINATION include/sep
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Create executables for each main file using add_sep_executable
add_sep_executable(trader_cli SOURCES app/cli_main.cpp)
target_link_libraries(trader_cli PRIVATE sep_lib)

add_sep_executable(data_downloader SOURCES app/data_downloader.cpp)
target_link_libraries(data_downloader PRIVATE sep_lib)

add_sep_executable(sep_dsl_interpreter SOURCES app/dsl_main.cpp)
target_link_libraries(sep_dsl_interpreter PRIVATE sep_lib)

add_sep_executable(quantum_tracker SOURCES app/quantum_tracker_main.cpp)
target_link_libraries(quantum_tracker PRIVATE sep_lib)

add_sep_executable(sep_app SOURCES app/app_main.cpp)
target_link_libraries(sep_app PRIVATE sep_lib)

add_sep_executable(quantum_pair_trainer SOURCES core/quantum_pair_trainer.cpp)
target_link_libraries(quantum_pair_trainer PRIVATE sep_lib)

add_sep_executable(sep SOURCES app/sep_main.cpp)
target_link_libraries(sep PRIVATE sep_lib)
