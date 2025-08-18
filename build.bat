@echo off
REM Windows build script for SEP Engine
REM Equivalent of build.sh for Windows development

setlocal EnableDelayedExpansion

echo Building SEP Engine on Windows...

REM Parse command line arguments
set REBUILD=false
set SKIP_DOCKER=false
set NATIVE_BUILD=true
set USE_CUDA=true

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--rebuild" set REBUILD=true
if "%~1"=="--no-docker" set SKIP_DOCKER=true
if "%~1"=="--no-cuda" set USE_CUDA=false
shift
goto parse_args
:args_done

REM Set default workspace path
if not defined SEP_WORKSPACE_PATH set SEP_WORKSPACE_PATH=%CD%

REM Create output directories
if not exist output mkdir output
if not exist build mkdir build

echo Building natively on Windows...

REM Clean up previous build artifacts if rebuild requested
if "%REBUILD%"=="true" (
    echo Performing a full rebuild...
    rmdir /s /q build >nul 2>&1
    rmdir /s /q output >nul 2>&1  
    del CMakeCache.txt >nul 2>&1
    del Makefile >nul 2>&1
    mkdir output
    mkdir build
)

cd build

REM Configure CUDA support
set CUDA_FLAGS=-DSEP_USE_CUDA=OFF
if "%USE_CUDA%"=="true" (
    where nvcc >nul 2>&1
    if !ERRORLEVEL! == 0 (
        echo CUDA detected, enabling CUDA support...
        set CUDA_FLAGS=-DSEP_USE_CUDA=ON
        
        REM Auto-detect CUDA_HOME if not set
        if not defined CUDA_HOME (
            for /f "tokens=*" %%i in ('where nvcc') do (
                set NVCC_PATH=%%i
                for %%j in ("!NVCC_PATH!") do set CUDA_HOME=%%~dpj
                set CUDA_HOME=!CUDA_HOME:~0,-5!
            )
            echo Auto-detected CUDA_HOME: !CUDA_HOME!
        ) else (
            echo Using existing CUDA_HOME: !CUDA_HOME!
        )
    ) else (
        echo CUDA not detected, building without CUDA support...
        set USE_CUDA=false
    )
)

REM Find Visual Studio Build Tools
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" 2>nul

REM Configure with CMake using MSVC
cmake .. -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE ^
    -DSEP_USE_GUI=OFF ^
    -DCMAKE_CXX_STANDARD=17 ^
    -DCMAKE_CXX_FLAGS="/W3 /EHsc /std:c++17" ^
    -DCMAKE_CUDA_STANDARD=17 ^
    %CUDA_FLAGS%

if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    exit /b 1
)

REM Build with Ninja
ninja -k 0 2>&1 | tee ..\output\build_log.txt

if errorlevel 1 (
    echo Build failed!
    cd ..
    exit /b 1
)

REM Copy compile_commands.json for IDE integration
copy compile_commands.json .. >nul 2>&1

cd ..

echo Build complete!
echo.
echo Built executables should be in build\ directory
echo Check output\build_log.txt for any warnings or errors