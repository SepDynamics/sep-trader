cmake_minimum_required(VERSION 3.18)

# Central test configuration for SEP
include(CTest)
include(${CMAKE_SOURCE_DIR}/cmake/gtest.cmake)

# Organize tests by type
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(performance)
