# DSL Test Executable
add_executable(dsl_test
    dsl_test.cpp
)

target_include_directories(dsl_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(dsl_test PRIVATE
    sep_dsl
)

target_compile_features(dsl_test PRIVATE cxx_std_17)

# DSL Core Primitives Test
add_executable(dsl_core_test
    dsl_core_test.cpp
)

target_include_directories(dsl_core_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(dsl_core_test PRIVATE
    sep_dsl
)

target_compile_features(dsl_core_test PRIVATE cxx_std_17)
