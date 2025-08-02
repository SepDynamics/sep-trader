# SEP DSL Library
add_library(sep_dsl STATIC
    # AST components
    ast/nodes.h
    
    # Parser components (will add .cpp as implemented)
    parser/parser.h
    parser/parser.cpp
    
    # Compiler components (will add .cpp as implemented)  
    compiler/compiler.h
    compiler/compiler.cpp
    
    # Runtime components (will add .cpp as implemented)
    runtime/runtime.h
    runtime/runtime.cpp
    
    # Standard library components (will add as implemented)
    # stdlib/patterns.h
    # stdlib/patterns.cpp
)

target_include_directories(sep_dsl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(sep_dsl PUBLIC
    sep_engine
    sep_quantum
    sep_memory
)

# Add compile features for C++17 (minimum for variant support in AST)
target_compile_features(sep_dsl PUBLIC cxx_std_17)
