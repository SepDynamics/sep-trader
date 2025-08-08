# DSL Library CMake Configuration

# Create the main DSL library
add_library(sep_dsl STATIC
    # AST nodes (header only)
    ast/nodes.h
    
    # AST serialization
    ast/serializer.h
    ast/serializer.cpp
    
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
    
    # Optimizer
    optimizer/optimizer.h
    optimizer/optimizer.cpp
)

# Enable Position Independent Code for shared library compatibility
set_target_properties(sep_dsl PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set include directories
target_include_directories(sep_dsl PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Set C++ standard
target_compile_features(sep_dsl PUBLIC cxx_std_17)

# Link against engine for builtin functions and nlohmann_json for serialization

# REMOVED: Problematic forced include

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

# Fixed by proper include order in standard_includes.h

# ENABLED: DSL interpreter executable
add_executable(sep_dsl_interpreter
    main.cpp
)

# REMOVED: Problematic forced include

# Link the interpreter executable with the DSL library and engine
target_link_libraries(sep_dsl_interpreter PRIVATE 
    sep_dsl 
    sep_engine 
    sep_trading 
    sep_core
    sep_config
    sep_cache
    sep_quantum
    sep_connectors
    nlohmann_json::nlohmann_json
    curl 
    hiredis 
    TBB::tbb 
    spdlog::spdlog 
    fmt::fmt 
    Threads::Threads 
    pqxx 
    pq
)
