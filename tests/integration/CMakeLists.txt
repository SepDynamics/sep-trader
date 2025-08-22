# Integration test targets
add_custom_target(run_integration_tests COMMENT "Aggregate target for integration tests")

# Create integration test executable
add_sep_test(integration_test_executable
    SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/../integration_test.cpp
    DEPENDENCIES
        CURL::libcurl
        nlohmann_json::nlohmann_json
)