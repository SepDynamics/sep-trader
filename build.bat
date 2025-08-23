@echo off
setlocal

echo Building SEP Engine with Docker on Windows...
echo Mounting local directory %CD% to /workspace in the container.

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

docker run --gpus all --rm -v "%CD%:/workspace" -w /workspace sep-trader-build bash -c "set -e; echo '--- Running build inside container ---'; git config --global --add safe.directory /workspace; mkdir -p build output; cd build; echo 'Configuring with CMake...'; cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE -DSEP_USE_CUDA=ON -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS='-Wno-pedantic -Wno-unknown-warning-option -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0' -DCMAKE_CUDA_FLAGS='-Wno-deprecated-gpu-targets -Xcompiler -Wno-pedantic -Xcompiler -Wno-unknown-warning-option -Xcompiler -Wno-invalid-source-encoding -D_GLIBCXX_USE_CXX11_ABI=0'; echo 'Building with Ninja...'; ninja -k 0; echo 'Copying compile_commands.json...'; cp compile_commands.json ..; echo '--- Build finished ---'" 2>&1 | powershell -Command "$input | Tee-Object -FilePath 'output\build_log.txt'"

set DOCKER_EXIT_CODE=%ERRORLEVEL%

REM Extract errors from build log like build.sh does
if exist "output\build_log.txt" (
    echo ---
    echo Extracting errors from build log...
    powershell -Command "Select-String -Path 'output\build_log.txt' -Pattern 'error|failed|fatal' -CaseSensitive:$false | ForEach-Object { $_.Line } | Out-File 'output\errors.txt' -Encoding UTF8"
    if exist "output\errors.txt" (
        for %%A in (output\errors.txt) do if %%~zA==0 (
            echo No errors found > output\errors.txt
        )
        echo Error summary saved to output\errors.txt
    )
)

if %DOCKER_EXIT_CODE% neq 0 (
    echo.
    echo Docker build failed!
    echo Check output\build_log.txt for detailed error information.
    echo Check output\errors.txt for filtered error summary.
    exit /b 1
)

echo.
echo Build complete!
echo Built executables should be in the build/ directory.
echo Build log saved to output\build_log.txt
echo Error summary saved to output\errors.txt