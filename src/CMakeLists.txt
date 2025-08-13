cmake_minimum_required(VERSION 3.18)
project(sep_trader)

add_definitions(-DSEP_BACKTESTING)
add_definitions(-DGLM_COMPILER=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# Set include directories
include_directories(
    ${CMAKE_BINARY_DIR}/_deps/spdlog-src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/app
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/io
    ${CMAKE_CURRENT_SOURCE_DIR}/util
)

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

# Create a library from the non-main source files
add_library(sep_lib ${LIB_SOURCES})

# Create executables for each main file
add_executable(trader_cli app/cli_main.cpp)
add_executable(oanda_trader app/oanda_trader_main.cpp)
add_executable(sep_app app/app_main.cpp)

# Link the executables to the library
target_link_libraries(trader_cli PRIVATE sep_lib CUDA::cudart)
target_link_libraries(oanda_trader PRIVATE sep_lib CUDA::cudart)
target_link_libraries(sep_app PRIVATE sep_lib CUDA::cudart)
