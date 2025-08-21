cmake_minimum_required(VERSION 3.18)

# Create test executable using SEP test framework
add_sep_test(sep_unit_tests
    SOURCES
        unit/coherence_manager_test.cpp
    DEPENDENCIES
        sep_lib
)