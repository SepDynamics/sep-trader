cmake_minimum_required(VERSION 3.18)

# Central test configuration for SEP
include(CTest)
include(${CMAKE_SOURCE_DIR}/cmake/gtest.cmake)

# Organize tests by type
add_subdirectory(unit)
add_subdirectory(app)
add_subdirectory(data_access)
add_subdirectory(dynamic_pair_manager)
add_subdirectory(market_cache)
add_subdirectory(pattern_metric_engine)
add_subdirectory(engine_config)
add_subdirectory(memory_tier_service)
add_subdirectory(processor)
add_subdirectory(facade)
