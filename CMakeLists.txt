cmake_minimum_required(VERSION 3.15)

# CRITICAL: Set compilers BEFORE project() to avoid CMake loops
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")
# set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")

# Set the module path to include our custom cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# Enable compile commands database generation for IDE support and static analysis
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# Project declaration with initial C++ support
project(sep_engine VERSION 1.0.0 LANGUAGES CXX)

# Ensure consistent C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    -Wall -Wextra -Wpedantic -O3
    -D_GLIBCXX_NO_ASSERT    # Fix functional header std::array issues
    -DPSTL_USE_TBB          # Enable proper TBB integration with PSTL
    -D_GLIBCXX_DEBUG_PEDANTIC=0  # Disable pedantic debug checks
    -D_GLIBCXX_CONCEPT_CHECKS=0  # Disable concept checks that might interfere
)


# Set up CUDA support
option(SEP_USE_CUDA "Enable CUDA support" OFF)
option(SEP_USE_GUI "Enable GUI support (ImGui)" OFF)
# Force GUI to be completely disabled for professional CLI system
set(SEP_USE_GUI OFF CACHE BOOL "GUI disabled for professional CLI trading system" FORCE)

if(SEP_USE_CUDA)
    # First verify CUDA paths
    if(NOT DEFINED ENV{CUDA_HOME})
        message(FATAL_ERROR "CUDA_HOME environment variable not set")
    endif()
    
    if(NOT EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
        message(FATAL_ERROR "NVCC not found at $ENV{CUDA_HOME}/bin/nvcc")
    endif()
    
    # Set CUDA paths before enabling language
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")
    set(CMAKE_CUDA_ARCHITECTURES 61;75;86;89)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    
    # Set CUDA flags BEFORE enabling language to ensure they're available during compiler identification
    set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler --extended-lambda --display-error-number")
    
    # Now enable CUDA language
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart nvml)
    message(STATUS "Found CUDAToolkit")
    message(STATUS "CUDA Libraries: ${CUDAToolkit_LIBRARIES}")
    
    # Print CUDA configuration for debugging
    message(STATUS "CUDA Home: $ENV{CUDA_HOME}")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # Add CUDA paths to include and link directories
    include_directories("$ENV{CUDA_HOME}/include")
    link_directories("$ENV{CUDA_HOME}/lib64")

endif()

# Include the template
include(cmake/template.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force compiler selection before project() call - move this to top
# This must be set before project() declaration to avoid CMake loops

# GLM configuration is included per-target in cmake template

# Enable testing
enable_testing()

# Find required packages
find_package(Threads REQUIRED)
# TBB built via FetchContent to avoid system dependencies
find_package(benchmark REQUIRED)
find_package(nlohmann_json REQUIRED)
include(cmake/gtest.cmake)
find_package(glm REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PostgreSQL REQUIRED)



include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

set_target_properties(yaml-cpp PROPERTIES CXX_STANDARD 17)

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


FetchContent_Declare(
  tbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG v2021.10.0
)
# TBB Configuration - Build requirements only  
set(TBB_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(TBB_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(TBB_TEST OFF CACHE BOOL "Build TBB tests" FORCE)
set(TBB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# Help find hwloc on Fedora systems (libraries in /usr/lib64)
link_directories(/usr/lib64)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
FetchContent_MakeAvailable(tbb)


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
        ${imgui_SOURCE_DIR}/backends
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
find_package(fmt REQUIRED)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.11.0
)
FetchContent_MakeAvailable(spdlog)

# Interface library for global includes
add_library(sep_global_includes INTERFACE)

# Temporarily disable global includes to isolate remaining errors
# target_compile_options(sep_global_includes INTERFACE 
#     "$<$<COMPILE_LANGUAGE:CXX>:-include;${CMAKE_CURRENT_SOURCE_DIR}/cmake/global_includes.h>"
#     "$<$<COMPILE_LANGUAGE:CUDA>:-include;${CMAKE_CURRENT_SOURCE_DIR}/cmake/global_includes.h>"
# )
target_link_libraries(sep_global_includes INTERFACE spdlog::spdlog nlohmann_json::nlohmann_json TBB::tbb)



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

add_subdirectory(extern/crow)
add_subdirectory(src)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_subdirectory(tests)
endif()
