@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo    Building CUDA Enterprise NPP Pipeline with NVCC (Windows x64)
echo ======================================================================

if not exist bin mkdir bin
if not exist output mkdir output
if not exist artifacts mkdir artifacts

set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo [!] Could not locate vcvars64.bat at %VCVARS%
    exit /b 1
)

echo [*] Initializing Visual Studio build environment...
call %VCVARS% >nul 2>&1

echo [*] Compiling C++ and CUDA sources with NVCC...
nvcc -std=c++17 -O3 ^
    -Iinclude -I"%CUDA_PATH%\include" ^
    src\main.cpp ^
    src\cli_parser.cpp ^
    src\image_io.cpp ^
    src\npp_pipeline.cpp ^
    src\pipeline_manager.cpp ^
    -L"%CUDA_PATH%\lib\x64" ^
    -lcudart -lnppc -lnppig -lnppif -lnppial -lnpps ^
    -o bin\cuda_enterprise_pipeline.exe

if %ERRORLEVEL% neq 0 (
    echo [!] Compilation failed with error code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo [OK] Build succeeded: bin\cuda_enterprise_pipeline.exe
exit /b 0
