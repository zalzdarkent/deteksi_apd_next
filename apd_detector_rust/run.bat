@echo off
REM APD Detector - Windows Batch Runner
REM Usage: run.bat [OPTIONS]

setlocal enabledelayedexpansion

echo.
echo ============================================
echo APD Detector - ONNX Runtime
echo ============================================
echo.

REM Check if Rust is installed
rustc --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Rust tidak terinstall!
    echo Download dari: https://www.rust-lang.org/tools/install
    pause
    exit /b 1
)

REM Check if model exists
if not exist "..\model\yolov26_retrain\best.onnx" (
    echo ERROR: Model tidak ditemukan!
    echo Expected: ..\model\yolov26_retrain\best.onnx
    pause
    exit /b 1
)

echo [✓] Rust terinstall
echo [✓] Model terdeteksi
echo.

REM Parse arguments
set BUILD_TYPE=release
set VERBOSE=

if "%1"=="debug" (
    set BUILD_TYPE=debug
    echo Mode: DEBUG (lebih lambat, more logging)
) else if "%1"=="release" (
    set BUILD_TYPE=release
    echo Mode: RELEASE (optimized, faster)
) else if "%1"=="clean" (
    echo Cleaning project...
    cargo clean
    exit /b 0
) else if "%1"=="help" (
    echo Usage: run.bat [OPTIONS]
    echo.
    echo OPTIONS:
    echo   debug       - Build dalam debug mode (default: release)
    echo   release     - Build dalam release mode (faster)
    echo   clean       - Hapus semua build files
    echo   help        - Show this message
    echo.
    exit /b 0
)

echo.
echo Building project...
echo.

if "%BUILD_TYPE%"=="debug" (
    cargo build
    if errorlevel 1 (
        echo Build failed!
        pause
        exit /b 1
    )
    echo.
    echo Running...
    .\target\debug\apd_detector.exe
) else (
    cargo build --release
    if errorlevel 1 (
        echo Build failed!
        pause
        exit /b 1
    )
    echo.
    echo Running...
    .\target\release\apd_detector.exe
)

if errorlevel 1 (
    echo.
    echo Execution failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Process completed!
echo Output: ..\output_videos\
echo ============================================
echo.
pause
