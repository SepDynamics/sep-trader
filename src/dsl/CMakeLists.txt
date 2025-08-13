add_sep_library(sep_dsl
    SOURCES
        ast/serializer.cpp
        lexer/lexer.cpp
        parser/parser.cpp
        runtime/interpreter.cpp
        runtime/runtime.cpp
        optimizer/optimizer.cpp
    HEADERS
        ast/nodes.h
        ast/serializer.h
        lexer/lexer.h
        parser/parser.h
        runtime/interpreter.h
        runtime/runtime.h
        optimizer/optimizer.h
    PCH_HEADER ../common/sep_precompiled.h
    DEPENDENCIES
        sep_engine
        nlohmann_json::nlohmann_json
)

target_include_directories(sep_dsl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${spdlog_SOURCE_DIR}/include
)