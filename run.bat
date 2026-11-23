@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo    Executing CUDA Enterprise Pipeline (Windows x64)
echo ======================================================================

set BIN_PATH=bin\cuda_enterprise_pipeline.exe

if not exist %BIN_PATH% (
    if exist build\Release\cuda_enterprise_pipeline.exe (
        copy /y build\Release\cuda_enterprise_pipeline.exe bin\cuda_enterprise_pipeline.exe
    ) else (
        echo [*] Binary not found. Building project first...
        call build.bat
        if %ERRORLEVEL% neq 0 exit /b 1
    )
)

if "%~1"=="" (
    echo [*] Running full enterprise batch workflow...
    %BIN_PATH% --input_dir data\input --output_dir output --streams 4 --operation all --angle 45.0
    echo [*] Running scaling and concurrency benchmark...
    %BIN_PATH% --input_dir data\input --output_dir output --benchmark
) else (
    %BIN_PATH% %*
)
