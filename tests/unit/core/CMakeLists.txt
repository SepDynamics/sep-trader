# Core engine unit tests go here

# Add coherence manager tests
add_sep_test(coherence_manager_test
    SOURCES coherence_manager_test.cpp
    DEPENDENCIES sep_lib
)

# Add result types tests
add_sep_test(result_types_test
    SOURCES result_types_test.cpp
    DEPENDENCIES sep_lib
)