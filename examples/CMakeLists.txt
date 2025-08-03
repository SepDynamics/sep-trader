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

# Weighted Sum Test
add_executable(weighted_sum_test
    weighted_sum_test.cpp
)

target_include_directories(weighted_sum_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(weighted_sum_test PRIVATE
    sep_dsl
)

target_compile_features(weighted_sum_test PRIVATE cxx_std_17)

# Pattern Inheritance Test
add_executable(pattern_inheritance_test
    pattern_inheritance_test.cpp
)

target_include_directories(pattern_inheritance_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(pattern_inheritance_test PRIVATE
    sep_dsl
)

target_compile_features(pattern_inheritance_test PRIVATE cxx_std_17)
