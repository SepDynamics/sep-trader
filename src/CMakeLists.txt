cmake_minimum_required(VERSION 3.18)
project(sep_trader)

add_definitions(-DSEP_BACKTESTING)
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
add_sep_library(sep_lib SOURCES ${LIB_SOURCES} DEPENDENCIES sep_core_deps CUDA::cudart PCH_HEADER core/sep_precompiled.h)
target_include_directories(sep_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/quantum/bitspace)

# Create executables for each main file using add_sep_executable
add_sep_executable(trader_cli SOURCES app/cli_main.cpp cuda/quantum_training.cu DEPENDENCIES sep_lib)
add_sep_executable(oanda_trader SOURCES app/oanda_trader_main.cpp DEPENDENCIES sep_lib)
add_sep_executable(data_downloader SOURCES app/data_downloader.cpp DEPENDENCIES sep_lib)
add_sep_executable(sep_dsl_interpreter SOURCES app/dsl_main.cpp DEPENDENCIES sep_lib)
add_sep_executable(quantum_tracker SOURCES app/quantum_tracker_main.cpp DEPENDENCIES sep_lib)
add_sep_executable(sep_app SOURCES app/app_main.cpp DEPENDENCIES sep_lib)
add_sep_executable(quantum_pair_trainer SOURCES core/quantum_pair_trainer.cpp DEPENDENCIES sep_lib)
