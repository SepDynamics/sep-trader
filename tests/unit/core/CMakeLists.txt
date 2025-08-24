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

# Dynamic config manager tests
add_sep_test(dynamic_config_manager_test
    SOURCES dynamic_config_manager_test.cpp
    DEPENDENCIES sep_lib
)

# CLI commands tests
add_sep_test(cli_commands_test
    SOURCES cli_commands_test.cpp
    DEPENDENCIES sep_lib
)

# Quantum metrics test
add_sep_test(quantum_metrics_test
    SOURCES quantum_metrics_test.cpp
    DEPENDENCIES sep_lib
)
# Placeholder for additional core tests
# add_sep_test(core_example_test
#     SOURCES core_example_test.cpp
#     DEPENDENCIES sep_lib
# )
