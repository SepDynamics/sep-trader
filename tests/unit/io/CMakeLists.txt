# IO library unit tests go here

# Add C API tests
add_sep_test(sep_c_api_test
    SOURCES sep_c_api_test.cpp
    DEPENDENCIES sep_lib
)
# Placeholder for additional IO tests
# add_sep_test(io_example_test
#     SOURCES io_example_test.cpp
#     DEPENDENCIES sep_lib
# )
