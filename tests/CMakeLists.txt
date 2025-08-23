cmake_minimum_required(VERSION 3.18)

# Central test configuration for SEP
include(CTest)
include(${CMAKE_SOURCE_DIR}/cmake/gtest.cmake)

# Organize tests by type
add_subdirectory(stdlib)
add_subdirectory(compiler)
add_subdirectory(interpreter)
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(performance)
add_subdirectory(app)
add_subdirectory(data_fetcher)
add_subdirectory(quantum_pair_trainer)
add_subdirectory(training_session)
add_subdirectory(pattern_evolution)
add_subdirectory(ticker_analyzer)
