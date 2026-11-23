@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo    Executing CUDA Enterprise NPP Pipeline (Windows x64)
echo ======================================================================

set BIN_PATH=bin\cuda_enterprise_pipeline.exe

if not exist %BIN_PATH% (
    echo [*] Binary not found. Building project first...
    call build.bat
    if %ERRORLEVEL% neq 0 exit /b 1
)

if "%~1"=="" (
    echo [*] Running full batch NPP rotation workflow across all 125 images...
    %BIN_PATH% --input_dir data\input --output_dir output --streams 4 --operation rotate --angle 45.0
    echo [*] Running multi-stream scaling benchmark...
    %BIN_PATH% --input_dir data\input --output_dir output --benchmark
) else (
    %BIN_PATH% %*
)
