#!/bin/bash
# APD Detector - Linux/Mac Runner

set -e

echo ""
echo "============================================"
echo "APD Detector - ONNX Runtime"
echo "============================================"
echo ""

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo "ERROR: Rust not installed!"
    echo "Install from: https://www.rust-lang.org/tools/install"
    exit 1
fi

# Check model
if [ ! -f "../model/yolov26_retrain/best.onnx" ]; then
    echo "ERROR: Model not found!"
    echo "Expected: ../model/yolov26_retrain/best.onnx"
    exit 1
fi

echo "[✓] Rust installed"
echo "[✓] Model found"
echo ""

# Parse arguments
BUILD_TYPE="release"

case "$1" in
    debug)
        BUILD_TYPE="debug"
        echo "Mode: DEBUG (slower, more logging)"
        ;;
    release)
        BUILD_TYPE="release"
        echo "Mode: RELEASE (optimized, faster)"
        ;;
    clean)
        echo "Cleaning project..."
        cargo clean
        exit 0
        ;;
    help)
        echo "Usage: ./run.sh [OPTIONS]"
        echo ""
        echo "OPTIONS:"
        echo "  debug   - Build in debug mode"
        echo "  release - Build in release mode (default, faster)"
        echo "  clean   - Remove all build files"
        echo "  help    - Show this message"
        echo ""
        exit 0
        ;;
esac

echo ""
echo "Building project..."
echo ""

if [ "$BUILD_TYPE" = "debug" ]; then
    cargo build
    ./target/debug/apd_detector
else
    cargo build --release
    ./target/release/apd_detector
fi

echo ""
echo "============================================"
echo "Process completed!"
echo "Output: ../output_videos/"
echo "============================================"
echo ""
