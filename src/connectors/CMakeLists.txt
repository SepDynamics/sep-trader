# Find required dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(CURL REQUIRED libcurl)

# Create the connectors library
add_library(sep_connectors STATIC
    oanda_connector.cpp
    market_data_converter.cpp
)

# Enable Position Independent Code for shared library compatibility
set_target_properties(sep_connectors PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(sep_connectors PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
PRIVATE
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(sep_connectors PUBLIC
    sep_engine
    ${CURL_LIBRARIES}
    nlohmann_json::nlohmann_json
    sep_global_includes
    TBB::tbb
)

target_compile_options(sep_connectors PRIVATE ${CURL_CFLAGS_OTHER})


