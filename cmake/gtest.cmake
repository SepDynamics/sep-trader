# Google Test Integration for SEP
# Provides testing framework setup

# Only set up GTest if testing is enabled
if(BUILD_TESTING)
    # Use system GTest instead of downloading
    find_package(GTest REQUIRED)
    
    # Include Google Test utilities
    include(GoogleTest)
    
    # Function to create SEP tests
    function(add_sep_test name)
        set(options "")
        set(oneValueArgs "")
        set(multiValueArgs SOURCES DEPENDENCIES)
        
        cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        
        # Create test executable
        add_executable(${name} ${ARG_SOURCES})
        
        # Link with GTest
        target_link_libraries(${name} PRIVATE
            GTest::gtest_main
            GTest::gtest
            ${ARG_DEPENDENCIES}
        )
        
        # Set include directories
        target_include_directories(${name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src
        )
        
        # Set C++17 standard
        target_compile_features(${name} PUBLIC cxx_std_17)
        
        # Discover tests
        gtest_discover_tests(${name})
        
        # Add to test suite
        add_test(NAME ${name} COMMAND ${name})
    endfunction()
    
    message(STATUS "Google Test enabled for SEP testing")
else()
    # Define empty function if testing is disabled
    function(add_sep_test name)
        # Do nothing
    endfunction()
    
    message(STATUS "Testing disabled - Google Test not configured")
endif()
