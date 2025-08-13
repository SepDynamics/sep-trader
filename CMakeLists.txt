cmake_minimum_required(VERSION 3.15)

# CRITICAL: Set compilers BEFORE project() to avoid CMake loops
set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-11")

# Set workspace path from environment or use current directory
if(DEFINED ENV{SEP_WORKSPACE_PATH})
    set(SEP_WORKSPACE_PATH $ENV{SEP_WORKSPACE_PATH})
else()
    set(SEP_WORKSPACE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Set the module path to include our custom cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "SEP_WORKSPACE_PATH: ${SEP_WORKSPACE_PATH}")

# Enable compile commands database generation for IDE support and static analysis
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# Project declaration with initial C++ support
project(sep_engine VERSION 1.0.0 LANGUAGES CXX)

# Ensure consistent C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable precompiled headers
set(CMAKE_DISABLE_PRECOMPILE_HEADERS OFF)

# GLOBAL FIX: Force include array header and fix std library issues
add_compile_definitions(
    _GLIBCXX_DEBUG_PEDANTIC=0  # Disable pedantic debug checks
    _GLIBCXX_CONCEPT_CHECKS=0  # Disable concept checks that might interfere
    GLM_FORCE_CUDA_VERSION=1100
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")

# CRITICAL FIX: Force include array header for all compilation units
# This ensures std::array is available for nlohmann_json and all dependencies
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include array") # Removed to avoid conflicts with std::array

# Add the C++ standard library include path to the global flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /usr/include/c++/11")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem /usr/include/c++/11")


# Removed force-include of array_protection.h to avoid breaking standard library dependencies


if(SEP_USE_TBB)
    add_compile_definitions(PSTL_USE_TBB)
endif()


# Set up CPU/CUDA configuration
option(SEP_CPU_ONLY "Build without CUDA and enable backtesting stubs" OFF)
option(SEP_USE_CUDA "Enable CUDA support" ON)
option(SEP_USE_GUI "Enable GUI support (ImGui)" OFF)
# Force GUI to be completely disabled for professional CLI system
set(SEP_USE_GUI OFF CACHE BOOL "GUI disabled for professional CLI trading system" FORCE)

if(SEP_CPU_ONLY)
    add_compile_definitions(SEP_BACKTESTING)
    set(SEP_USE_CUDA OFF CACHE BOOL "Disable CUDA when CPU-only" FORCE)
endif()

if(SEP_USE_CUDA)
    # First verify CUDA paths
    if(NOT DEFINED ENV{CUDA_HOME})
        message(FATAL_ERROR "CUDA_HOME environment variable not set")
    endif()
    
    if(NOT EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
        message(FATAL_ERROR "NVCC not found at $ENV{CUDA_HOME}/bin/nvcc")
    endif()
    
    # Set CUDA paths and flags before enabling language
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-11")
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures")
    
    # CUDA flags are now set per-target in template.cmake
    
    # Configure CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    
    # Enable CUDA language and find toolkit
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Removed force-include of array_protection.h for CUDA to avoid breaking standard library
    
    message(STATUS "Found CUDAToolkit")
    message(STATUS "CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")
    
    # Explicitly set CUDA libraries
    set(CUDA_LIBRARIES
        CUDA::cudart
        CUDA::cuda_driver
    )
    
    # Print CUDA configuration for debugging
    message(STATUS "CUDA Home: $ENV{CUDA_HOME}")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # Add CUDA paths to include and link directories
    include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    include_directories(SYSTEM /usr/include/c++/11)
    link_directories(${CUDAToolkit_LIBRARY_DIR})

endif()

# Fix fmt/spdlog conflicts - use external fmt
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt library")

# Include the template
include(cmake/template.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C++ compilation flags with forced std headers for nlohmann compatibility


# The following global include is problematic and has been removed.
# Include directories should be added on a per-target basis.
include_directories(${CMAKE_SOURCE_DIR}/src)

# Disable nlohmann global UDLs to prevent conflicts
add_compile_definitions(NLOHMANN_JSON_USE_GLOBAL_UDLS=0)

# Force compiler selection before project() call - move this to top
# This must be set before project() declaration to avoid CMake loops

# GLM configuration is included per-target in cmake template

# Enable testing
enable_testing()

# Find required packages
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)
# TBB built via FetchContent to avoid system dependencies
find_package(benchmark)
include(cmake/gtest.cmake)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PQXX REQUIRED libpqxx)
pkg_check_modules(PQ REQUIRED libpq)

if(PQXX_FOUND AND PQ_FOUND)
    message(STATUS "Found PostgreSQL libraries using pkg-config")
    add_library(PostgreSQL::pqxx INTERFACE IMPORTED)
    set_property(TARGET PostgreSQL::pqxx PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${PQXX_INCLUDE_DIRS}")
    set_property(TARGET PostgreSQL::pqxx PROPERTY INTERFACE_LINK_LIBRARIES "${PQXX_LIBRARIES}")
    add_library(PostgreSQL::pq INTERFACE IMPORTED)
    set_property(TARGET PostgreSQL::pq PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${PQ_INCLUDE_DIRS}")
    set_property(TARGET PostgreSQL::pq PROPERTY INTERFACE_LINK_LIBRARIES "${PQ_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find PostgreSQL libraries using pkg-config")
endif()

include(FetchContent)

# Use system GLM instead of downloading
find_package(glm REQUIRED)
# Use FetchContent for nlohmann_json with GCC 11 fix
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Apply surgical fix for std::array visibility to nlohmann_json
# Use the base target, not the alias



# Use system yaml-cpp instead of downloading
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

if(YAML_CPP_FOUND)
    add_library(YAML::yaml-cpp INTERFACE IMPORTED)
    set_property(TARGET YAML::yaml-cpp PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIRS}")
    set_property(TARGET YAML::yaml-cpp PROPERTY INTERFACE_LINK_LIBRARIES "${YAML_CPP_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not find yaml-cpp library using pkg-config")
endif()

if(SEP_USE_GUI)
    FetchContent_Declare(
      imgui
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG v1.91.6
    )
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(
      implot
      GIT_REPOSITORY https://github.com/epezent/implot.git
      GIT_TAG v0.16
    )
    FetchContent_MakeAvailable(implot)
endif()


# Use FetchContent to build TBB and avoid system dependencies
option(SEP_USE_TBB "Enable TBB support" ON)
if(SEP_USE_TBB)
    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
    FetchContent_Declare(
        oneTBB
        GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
        GIT_TAG v2021.11.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(oneTBB)
endif()


if(SEP_USE_GUI)
    # Create imgui library manually since imgui doesn't provide CMakeLists.txt
    set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )

    add_library(imgui STATIC ${IMGUI_SOURCES})

    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgai_SOURCE_DIR}/backends
    )

    set(IMPLOT_SOURCES
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
        ${implot_SOURCE_DIR}/implot_demo.cpp
    )

    add_library(implot STATIC ${IMPLOT_SOURCES})

    target_include_directories(implot PUBLIC
        ${implot_SOURCE_DIR}
    )

    target_link_libraries(implot PUBLIC imgui)

    # Find and link required dependencies
    find_package(PkgConfig REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)

    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
endif()

# Add subdirectories
# Use FetchContent for fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.2.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fmt)

# Use FetchContent for spdlog
add_definitions(-DSPDLOG_FMT_EXTERNAL)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

add_library(sep_core_deps INTERFACE)
target_include_directories(sep_core_deps INTERFACE 
    ${spdlog_SOURCE_DIR}/include
    ${fmt_SOURCE_DIR}/include
)
target_link_libraries(sep_core_deps INTERFACE
    Threads::Threads
#    benchmark::benchmark
#    GTest::gtest
#    GTest::gmock
    PostgreSQL::pqxx
    PostgreSQL::pq
    glm::glm
    nlohmann_json::nlohmann_json
    YAML::yaml-cpp
    TBB::tbb
    fmt::fmt
    spdlog::spdlog
    CURL::libcurl
)



# Enable fuzzing if explicitly requested
option(ENABLE_FUZZING "Enable fuzz testing with LibFuzzer" OFF)
if(ENABLE_FUZZING)
    add_subdirectory(tests/fuzzing)
endif()

# Enable memory testing if explicitly requested
option(ENABLE_MEMORY_TESTING "Enable memory leak detection with AddressSanitizer/Valgrind" OFF)
if(ENABLE_MEMORY_TESTING)
    add_subdirectory(tests/memory)
endif()

# Custom command to generate documentation
add_custom_target(generate_docs
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/parse_docs.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating documentation from source headers"
)

add_subdirectory(src)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_subdirectory(tests)
endif()