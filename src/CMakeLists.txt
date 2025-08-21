cmake_minimum_required(VERSION 3.18)
project(sep_trader)

add_definitions(-DGLM_COMPILER=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# Gather all source files
file(GLOB_RECURSE ALL_SOURCES
    "app/*.cpp"
    "app/*.cu"
    "core/*.cpp"
    "core/*.cu"
    "cuda/*.cpp"
    "cuda/*.cu"
    "io/*.cpp"
    "io/*.cu"
    "quantum/bitspace/*.cpp"
    "quantum/bitspace/*.cu"
    "util/*.cpp"
    "util/*.cu"
)

# Separate main files from library source files
set(LIB_SOURCES)
foreach(file ${ALL_SOURCES})
    if(NOT file MATCHES "main.cpp$")
        list(APPEND LIB_SOURCES ${file})
    endif()
endforeach()

# Create a library from the non-main source files using add_sep_library
add_sep_library(sep_lib SOURCES ${LIB_SOURCES} PCH_HEADER core/sep_precompiled.h)
target_link_libraries(sep_lib
    PUBLIC
        sep_core_deps
        CUDA::cudart
)
target_include_directories(sep_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/quantum/bitspace)

# Collect public headers for installation
file(GLOB_RECURSE SEP_LIB_PUBLIC_HEADERS
    "app/*.h" "app/*.hpp"
    "core/*.h" "core/*.hpp"
    "cuda/*.h" "cuda/*.hpp"
    "io/*.h" "io/*.hpp"
    "quantum/bitspace/*.h" "quantum/bitspace/*.hpp"
    "util/*.h" "util/*.hpp"
)
set_target_properties(sep_lib PROPERTIES PUBLIC_HEADER "${SEP_LIB_PUBLIC_HEADERS}")

install(TARGETS sep_lib
    PUBLIC_HEADER DESTINATION include/sep
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Create executables for each main file using add_sep_executable
add_sep_executable(trader_cli SOURCES app/cli_main.cpp cuda/quantum_training.cu)
target_link_libraries(trader_cli PRIVATE sep_lib)

add_sep_executable(oanda_trader SOURCES app/oanda_trader_main.cpp)
target_link_libraries(oanda_trader PRIVATE sep_lib)

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
