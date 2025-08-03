# DSL Library CMake Configuration

# Create the main DSL library
add_library(sep_dsl STATIC
    # AST nodes (header only)
    ast/nodes.h
    
    # Lexer
    lexer/lexer.h
    lexer/lexer.cpp
    
    # Parser 
    parser/parser.h
    parser/parser.cpp
    
    # Runtime (tree-walk interpreter)
    runtime/interpreter.h
    runtime/interpreter.cpp
    runtime/runtime.h
    runtime/runtime.cpp
)

# Set include directories
target_include_directories(sep_dsl PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Set C++ standard
target_compile_features(sep_dsl PUBLIC cxx_std_17)

# Link against engine for builtin functions
target_link_libraries(sep_dsl PRIVATE sep_engine)

# Add compiler flags
target_compile_options(sep_dsl PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Optional: Add debug information in debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sep_dsl PRIVATE -g -O0)
else()
    target_compile_options(sep_dsl PRIVATE -O2)
endif()

# Create the DSL interpreter executable
add_executable(sep_dsl_interpreter
    main.cpp
)

# Link the interpreter executable with the DSL library and engine
target_link_libraries(sep_dsl_interpreter PRIVATE sep_dsl sep_engine)
