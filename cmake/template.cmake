# SEP Template CMake Functions
# Provides standardized library and executable creation

# CUDA settings are configured globally in root CMakeLists.txt

# Function to create a standard SEP library
function(add_sep_library name)
    # Enable CUDA language first if needed
    if(SEP_USE_CUDA)
        enable_language(CUDA)
    endif()

    set(options STATIC SHARED)
    set(oneValueArgs PCH_HEADER)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES CUDA_SOURCES)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Default to STATIC if not specified
    if(NOT ARG_STATIC AND NOT ARG_SHARED)
        set(ARG_STATIC TRUE)
    endif()
    
    # Combine regular sources and CUDA sources
    set(ALL_SOURCES ${ARG_SOURCES} ${ARG_CUDA_SOURCES})
    
    # Create library
    if(ARG_STATIC)
        add_library(${name} STATIC ${ALL_SOURCES} ${ARG_HEADERS})
    else()
        add_library(${name} SHARED ${ALL_SOURCES} ${ARG_HEADERS})
    endif()
    
    # Set CUDA properties and flags if CUDA sources are present
    if(ARG_CUDA_SOURCES AND SEP_USE_CUDA)
        set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        
        # Group CUDA headers separately
        source_group("CUDA Headers" FILES ${ARG_HEADERS})
        foreach(header ${ARG_HEADERS})
            get_filename_component(ext ${header} EXT)
            if(ext STREQUAL ".cuh")
                set_source_files_properties(${header} PROPERTIES
                    HEADER_FILE_ONLY ON
                )
            endif()
        endforeach()
        
        # Apply CUDA compilation flags to the target
        set(CUDA_FLAGS "--expt-relaxed-constexpr --extended-lambda")
        foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
            string(APPEND CUDA_FLAGS " --generate-code=arch=compute_${arch},code=[compute_${arch},sm_${arch}]")
        endforeach()
        
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
    endif()
    
    # Set include directories
    target_include_directories(${name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
    )
    
    # Add dependencies (filter out CUDA when disabled)
    if(ARG_DEPENDENCIES)
        set(FILTERED_DEPS "")
        foreach(dep ${ARG_DEPENDENCIES})
            if(dep MATCHES "CUDA::" AND NOT SEP_USE_CUDA)
                # Skip CUDA dependencies when CUDA is disabled
                continue()
            endif()
            list(APPEND FILTERED_DEPS ${dep})
        endforeach()
        if(FILTERED_DEPS)
            target_link_libraries(${name} PUBLIC ${FILTERED_DEPS})
        endif()
    endif()
    
    # Set C++17 standard
    target_compile_features(${name} PUBLIC cxx_std_17)

    if(ARG_PCH_HEADER)
        target_precompile_headers(${name} PUBLIC ${ARG_PCH_HEADER})
    endif()
    
    # Set position independent code for all libraries (needed for shared libraries)
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    # Common compile options
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${name} PRIVATE 
            -Wall 
            -Wextra 
            -Wpedantic
            -O3
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${name} PRIVATE 
            /W4
            /O2
        )
    endif()
endfunction()

# Function to create a standard SEP executable
function(add_sep_executable name)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES DEPENDENCIES)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Create executable
    add_executable(${name} ${ARG_SOURCES})
    
    # Set include directories
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
    )
    
    # Link required core dependencies and any additional libraries
    target_link_libraries(${name} PRIVATE sep_core_deps)
    if(ARG_DEPENDENCIES)
        target_link_libraries(${name} PRIVATE ${ARG_DEPENDENCIES})
    endif()
    
    # Set C++17 standard
    target_compile_features(${name} PUBLIC cxx_std_17)
    
    # Common compile options
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${name} PRIVATE 
            -Wall 
            -Wextra 
            -Wpedantic
            -O3
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${name} PRIVATE 
            /W4
            /O2
        )
    endif()
endfunction()
