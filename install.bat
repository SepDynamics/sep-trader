@echo off
REM Windows dependency installer for SEP Engine
REM Equivalent of install.sh for Windows development

setlocal EnableDelayedExpansion

echo Installing SEP Engine dependencies on Windows...

REM Parse command line arguments
set USE_CUDA=true
set USE_MINIMAL=false
set SKIP_VCPKG_BOOTSTRAP=false

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--no-cuda" set USE_CUDA=false
if "%~1"=="--minimal" set USE_MINIMAL=true  
if "%~1"=="--skip-vcpkg-bootstrap" set SKIP_VCPKG_BOOTSTRAP=true
shift
goto parse_args
:args_done

REM Create directories
if not exist logs mkdir logs
if not exist build\deps mkdir build\deps

echo.
echo === Installing System Dependencies ===

REM Install PostgreSQL
winget install PostgreSQL.PostgreSQL --accept-package-agreements --accept-source-agreements
if errorlevel 1 echo Warning: PostgreSQL installation may have failed

REM Install Redis (Windows version)
winget install Redis.Redis --accept-package-agreements --accept-source-agreements
if errorlevel 1 echo Warning: Redis installation may have failed

echo.
echo === Setting up vcpkg (C++ Package Manager) ===

REM Clone vcpkg if it doesn't exist
if not exist vcpkg (
    echo Cloning vcpkg...
    REM Use a fallback since git might not be in PATH yet
    where git >nul 2>&1
    if errorlevel 1 (
        echo Git not found in PATH, please install Git first and restart your terminal
        exit /b 1
    )
    git clone https://github.com/Microsoft/vcpkg.git
    if errorlevel 1 (
        echo Failed to clone vcpkg
        exit /b 1
    )
)

cd vcpkg

REM Bootstrap vcpkg if not already done
if not exist vcpkg.exe (
    if "%SKIP_VCPKG_BOOTSTRAP%"=="false" (
        echo Bootstrapping vcpkg...
        call bootstrap-vcpkg.bat
        if errorlevel 1 (
            echo vcpkg bootstrap failed
            cd ..
            exit /b 1
        )
    )
)

REM Integrate vcpkg with Visual Studio
vcpkg integrate install

echo.
echo === Installing C++ Dependencies via vcpkg ===

REM Define package lists
set MINIMAL_PACKAGES=curl nlohmann-json yaml-cpp fmt spdlog glm
set FULL_PACKAGES=%MINIMAL_PACKAGES% benchmark gtest libpqxx hiredis tbb

if "%USE_MINIMAL%"=="true" (
    set PACKAGES=%MINIMAL_PACKAGES%
    echo Installing minimal package set...
) else (
    set PACKAGES=%FULL_PACKAGES%  
    echo Installing full package set...
)

REM Install packages one by one with error checking
for %%p in (%PACKAGES%) do (
    echo Installing %%p...
    vcpkg install %%p:x64-windows
    if errorlevel 1 (
        echo Warning: Failed to install %%p, continuing...
        echo Failed to install %%p >> ..\logs\vcpkg_failures.txt
    )
)

REM Install CUDA-specific packages if CUDA is enabled  
if "%USE_CUDA%"=="true" (
    echo Installing CUDA-related packages...
    vcpkg install cuda:x64-windows
    if errorlevel 1 echo Warning: CUDA vcpkg package installation failed
)

cd ..

echo.
echo === Verifying Installations ===

REM Check system tools
where cmake >nul 2>&1 && echo CMake: OK || echo CMake: MISSING
where ninja >nul 2>&1 && echo Ninja: OK || echo Ninja: MISSING

if "%USE_CUDA%"=="true" (
    where nvcc >nul 2>&1 && echo NVCC: OK || echo NVCC: MISSING
)

REM Check vcpkg packages
echo Checking vcpkg packages...
vcpkg\vcpkg list > logs\vcpkg_installed.txt 2>&1

echo.
echo === Setup Complete ===
echo.
echo Next steps:
echo 1. Restart your terminal to ensure all PATH changes take effect
echo 2. Run: build.bat
echo 3. If build fails, check logs\build_log.txt for errors
echo.
echo Dependencies installed to:
echo - vcpkg packages: %CD%\vcpkg\installed\x64-windows\  
echo - PostgreSQL: System installation
echo - Redis: System installation
echo.
echo To build: build.bat
echo To clean build: build.bat --rebuild
echo To build without CUDA: build.bat --no-cuda