cmake_minimum_required(VERSION 3.15)

# Fewer surprise warnings from third-party junk
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Suppress deprecation warnings" FORCE)

# Compilers BEFORE project(): lock GCC-14 if present; otherwise allow nvcc to cope later
if(UNIX AND NOT APPLE)
  if(EXISTS "/usr/bin/gcc-14")
    set(CMAKE_C_COMPILER        "/usr/bin/gcc-14")
    set(CMAKE_CXX_COMPILER      "/usr/bin/g++-14")
    # Host compiler for CUDA will be respected later if CUDA is enabled
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-14")
    message(STATUS "Using GCC-14 on Linux (CUDA 12.9 compatible)")
  else()
    list(APPEND CMAKE_CUDA_FLAGS "--allow-unsupported-compiler")
    message(STATUS "GCC-14 not found; allowing unsupported compiler for CUDA toolchain")
  endif()
endif()

# Workspace and module path
if(DEFINED ENV{SEP_WORKSPACE_PATH})
  set(SEP_WORKSPACE_PATH "$ENV{SEP_WORKSPACE_PATH}")
else()
  set(SEP_WORKSPACE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "SEP_WORKSPACE_PATH: ${SEP_WORKSPACE_PATH}")

# Compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

project(sep_engine VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

# Provide project root path to C++ sources
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Common directories used across the project
set(CACHE_DIR "${CMAKE_SOURCE_DIR}/cache/")
set(CONFIG_DIR "${CMAKE_SOURCE_DIR}/config/")
set(LOG_DIR "${CMAKE_SOURCE_DIR}/logs/")

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Minimal, targeted defines
add_compile_definitions(
  _GLIBCXX_DEBUG_PEDANTIC=0
  _GLIBCXX_CONCEPT_CHECKS=0
  GLM_FORCE_CUDA_VERSION=1100
)

add_compile_options(-Wall -Wextra -Wpedantic -O3)

# If you actually use TBB in algorithms, this is the only PSTL hint you need
if(SEP_USE_TBB)
  add_compile_definitions(PSTL_USE_TBB)
endif()

# Runtime options
option(SEP_USE_CUDA "Enable CUDA support" ON)

# CUDA ENABLED - Professional Trading System with GPU acceleration
if(SEP_USE_CUDA)
    add_compile_definitions(SEP_USE_CUDA SEP_ENGINE_HAS_CUDA)

    if(NOT DEFINED ENV{CUDA_HOME})
        message(FATAL_ERROR "CUDA_HOME environment variable not set")
    endif()
    if(NOT EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
        message(FATAL_ERROR "NVCC not found at $ENV{CUDA_HOME}/bin/nvcc")
    endif()

    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
    if(UNIX AND NOT APPLE AND EXISTS "/usr/bin/g++-14")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-14")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "61;86;89" CACHE STRING "Supported CUDA architectures")

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Core compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda -Xcompiler -fno-strict-aliasing -Xcompiler -Wno-deprecated-declarations -Xcompiler -Wno-deprecated-copy -Xcompiler -D_FORCE_INLINES -Xcompiler -w")

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA Home: $ENV{CUDA_HOME}")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")

    set(CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver CUDA::cufft)
    include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    if(UNIX AND NOT APPLE AND EXISTS "/usr/include/c++/14")
        include_directories(SYSTEM /usr/include/c++/14)
    endif()
    link_directories(${CUDAToolkit_LIBRARY_DIR})
endif()

# fmt/spdlog: force spdlog to use external fmt
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt" FORCE)

include(cmake/template.cmake)

add_compile_definitions(NLOHMANN_JSON_USE_GLOBAL_UDLS=0)

option(BUILD_TESTING "Enable building of tests" ON)
include(CTest)
if(BUILD_TESTING)
  enable_testing()
  include(cmake/gtest.cmake)
endif()

find_package(CURL REQUIRED)
find_package(Threads REQUIRED)
find_package(benchmark QUIET)

# PostgreSQL (libpqxx + libpq)
if(WIN32)
  find_package(libpqxx CONFIG REQUIRED)
  find_package(unofficial-libpq CONFIG REQUIRED)
  add_library(PostgreSQL::pqxx ALIAS libpqxx::pqxx)
  add_library(PostgreSQL::pq   ALIAS unofficial::libpq::libpq)
  message(STATUS "PostgreSQL via vcpkg on Windows")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PQXX REQUIRED libpqxx)
  pkg_check_modules(PQ   REQUIRED libpq)

  add_library(PostgreSQL::pqxx INTERFACE IMPORTED)
  set_property(TARGET PostgreSQL::pqxx PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${PQXX_INCLUDE_DIRS}")
  set_property(TARGET PostgreSQL::pqxx PROPERTY INTERFACE_LINK_LIBRARIES "${PQXX_LIBRARIES}")

  add_library(PostgreSQL::pq INTERFACE IMPORTED)
  set_property(TARGET PostgreSQL::pq PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${PQ_INCLUDE_DIRS}")
  set_property(TARGET PostgreSQL::pq PROPERTY INTERFACE_LINK_LIBRARIES "${PQ_LIBRARIES}")

  message(STATUS "PostgreSQL via pkg-config on Linux")
endif()

include(FetchContent)

find_package(glm REQUIRED)

find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# Handle both system package and FetchContent target names
if(TARGET nlohmann_json::nlohmann_json AND NOT TARGET nlohmann_json)
  add_library(nlohmann_json ALIAS nlohmann_json::nlohmann_json)
endif()

# ---------- yaml-cpp ----------
if(WIN32)
  find_package(yaml-cpp CONFIG REQUIRED)
  add_library(YAML::yaml-cpp ALIAS yaml-cpp)
  message(STATUS "yaml-cpp: vcpkg (Windows)")
else()
  find_package(yaml-cpp QUIET)
  if(NOT yaml-cpp_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
    add_library(YAML::yaml-cpp INTERFACE IMPORTED)
    set_property(TARGET YAML::yaml-cpp PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIRS}")
    set_property(TARGET YAML::yaml-cpp PROPERTY INTERFACE_LINK_LIBRARIES     "${YAML_CPP_LIBRARIES}")
  else()
    # Handle different target names from system packages
    if(TARGET yaml-cpp AND NOT TARGET YAML::yaml-cpp)
      add_library(YAML::yaml-cpp ALIAS yaml-cpp)
    elseif(TARGET yaml-cpp::yaml-cpp AND NOT TARGET YAML::yaml-cpp)
      add_library(YAML::yaml-cpp ALIAS yaml-cpp::yaml-cpp)
    endif()
  endif()
endif()

# ---------- hiredis ----------
if(WIN32)
  find_package(hiredis CONFIG REQUIRED)
  add_library(Redis::hiredis ALIAS hiredis::hiredis)
  message(STATUS "hiredis: vcpkg (Windows)")
else()
  find_package(hiredis QUIET CONFIG)
  if(hiredis_FOUND)
    add_library(Redis::hiredis ALIAS hiredis::hiredis)
  else()
    if(NOT DEFINED PKG_CONFIG_FOUND)
      find_package(PkgConfig REQUIRED)
    endif()
    pkg_check_modules(HIREDIS REQUIRED hiredis)
    add_library(Redis::hiredis INTERFACE IMPORTED)
    set_property(TARGET Redis::hiredis PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${HIREDIS_INCLUDE_DIRS}")
    set_property(TARGET Redis::hiredis PROPERTY INTERFACE_LINK_LIBRARIES     "${HIREDIS_LIBRARIES}")
    message(STATUS "hiredis: pkg-config (Linux)")
  endif()
endif()

# ---------- oneTBB (optional) ----------
option(SEP_USE_TBB "Enable TBB support" ON)
if(SEP_USE_TBB)
  find_package(TBB QUIET)
  if(NOT TBB_FOUND)
    include(FetchContent)
    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
    FetchContent_Declare(
      oneTBB
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG v2021.11.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(oneTBB)
    if(NOT TARGET TBB::tbb AND TARGET tbb)
      add_library(TBB::tbb ALIAS tbb)
    endif()
  endif()
endif()

# ---------- fmt / spdlog ----------
find_package(fmt 11 QUIET)
if(NOT fmt_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(fmt)
endif()

find_package(spdlog 1.15 QUIET)
if(NOT spdlog_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(spdlog)
endif()

# ---------- Core dependency umbrella ----------
add_library(sep_core_deps INTERFACE)

foreach(_t
  Threads::Threads
  CURL::libcurl
  glm::glm
  PostgreSQL::pqxx
  PostgreSQL::pq
  YAML::yaml-cpp
  Redis::hiredis
  TBB::tbb
  fmt::fmt
  spdlog::spdlog
  benchmark::benchmark
  GTest::gtest
  GTest::gmock
)
  if(TARGET ${_t})
    target_link_libraries(sep_core_deps INTERFACE ${_t})
  endif()
endforeach()

add_library(sep_fetchcontent_deps INTERFACE)
if(TARGET spdlog)
  get_target_property(_spdlog_inc spdlog INTERFACE_INCLUDE_DIRECTORIES)
  if(_spdlog_inc)
    target_include_directories(sep_fetchcontent_deps INTERFACE ${_spdlog_inc})
  endif()
endif()
if(TARGET fmt)
  get_target_property(_fmt_inc fmt INTERFACE_INCLUDE_DIRECTORIES)
  if(_fmt_inc)
    target_include_directories(sep_fetchcontent_deps INTERFACE ${_fmt_inc})
  endif()
endif()
target_link_libraries(sep_fetchcontent_deps INTERFACE
  $<$<TARGET_EXISTS:nlohmann_json::nlohmann_json>:nlohmann_json::nlohmann_json>
  $<$<TARGET_EXISTS:fmt::fmt>:fmt::fmt>
  $<$<TARGET_EXISTS:spdlog::spdlog>:spdlog::spdlog>
)

# ---------- Tests & subdirs ----------
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
add_subdirectory(src)

# ---------- Install ----------
# Note: This is a standalone application, not a library for external consumption
