# Setup Guide - APD Detector Rust

## Prerequisites

### Windows 10/11

#### 1. Install Rust

```powershell
# Download dan jalankan rustup installer
irm https://sh.rustup.rs | iex

# Verify installation
rustc --version
cargo --version
```

#### 2. Install MSVC Build Tools (Required)

Download dari: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Atau via package manager:
```powershell
# Menggunakan Chocolatey (jika terinstall)
choco install visualstudio2019-workload-vctools

# Atau via winget (Windows 11)
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools"
```

#### 3. Install OpenCV

**Option A: Using vcpkg (Recommended)**

```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Install OpenCV
.\vcpkg install opencv:x64-windows

# Set environment variable
$env:OpenCV_DIR = "C:\path\to\vcpkg\installed\x64-windows"

# Atau add ke system environment variables untuk permanent
[Environment]::SetEnvironmentVariable("OpenCV_DIR", "C:\path\to\vcpkg\installed\x64-windows", "User")
```

**Option B: Manual Build**

1. Download OpenCV source: https://github.com/opencv/opencv/releases
2. Build dengan CMake:
```powershell
# Download source
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build

# Configure with CMake
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build
cmake --build . --config Release --parallel 4

# Install
cmake --install . --config Release --prefix "C:\opencv"

# Set environment variable
[Environment]::SetEnvironmentVariable("OpenCV_DIR", "C:\opencv", "User")
```

#### 4. Verify OpenCV Installation

```powershell
# Test OpenCV libraries accessible
ls $env:OpenCV_DIR\lib

# Should show something like:
# opencv_core.lib
# opencv_imgproc.lib
# opencv_videoio.lib
# etc.
```

## Building APD Detector

### 1. Navigate ke Project Directory

```powershell
cd c:\Magang_Alif\Trial\apd_detector_rust
```

### 2. Build Project

```powershell
# Debug build (cepat untuk development)
cargo build

# Release build (optimized untuk production)
cargo build --release
```

First build akan cukup lama karena men-download dan compile dependencies:
- ONNX Runtime (~5-10 menit)
- OpenCV Rust bindings (~3-5 menit)
- Other dependencies (~2-3 menit)

**Total: ~15-20 menit untuk first build**

### 3. Troubleshooting Build

#### Error: "OpenCV not found"

Solution:
```powershell
# Set path ke OpenCV installation
$env:OpenCV_DIR = "C:\path\to\opencv"

# Atau add ke environment permanently
[Environment]::SetEnvironmentVariable("OpenCV_DIR", "C:\path\to\opencv", "User")

# Restart terminal dan try again
cargo clean
cargo build --release
```

#### Error: "MSVC compiler not found"

Solution:
```powershell
# Install Visual C++ build tools:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Atau use rustup untuk switch toolchain
rustup default stable-msvc
```

#### Error: "ort" crate compilation fails

Solution:
```powershell
# First time ONNX Runtime di-compile, butuh internet connection
# Pastikan firewall/antivirus tidak blocking download

# Try clean build:
cargo clean
cargo build --release
```

## Running APD Detector

### 1. Prepare Model

Pastikan model ada di:
```
c:\Magang_Alif\Trial\model\yolov26_retrain\best.onnx
```

### 2. Prepare Video Files

Pastikan video ada di:
```
c:\Magang_Alif\Trial\samples\
```

Video formats supported: mp4, avi, mov, mkv, flv, wmv

### 3. Running

**Debug version (slower):**
```powershell
cargo run
```

**Release version (faster, recommended):**
```powershell
cargo run --release
```

Atau jalankan binary directly:
```powershell
.\target\release\apd_detector.exe
```

### 4. Output

Video dengan detections akan tersimpan di:
```
c:\Magang_Alif\Trial\output_videos\
```

Contoh output file:
```
detected_kaizen_casting.mp4
detected_workshop_kaizen.mp4
etc.
```

## Optimizations

### For faster builds:

Add ke `Cargo.toml`:
```toml
[profile.dev]
opt-level = 1  # Slightly optimized debug builds
```

### For smaller binary:

```powershell
cargo build --release
```

Binary size: ~50-100 MB (termasuk semua dependencies)

### Strip debug symbols (optional):

```powershell
# After build
strip .\target\release\apd_detector.exe
```

## Verifying Installation

### Check Rust version:
```powershell
rustc --version
cargo --version
```

Expected output:
```
rustc 1.xx.x (xxxxx xxxxx)
cargo 1.xx.x
```

### Check ONNX Runtime:
Binary akan di-download di first run, verify dengan check:
```powershell
file .\target\release\apd_detector.exe
```

### Quick Test:

Jalankan dengan single video:

Modifikasi `src/main.rs`:
```rust
// Add ini untuk test single video:
println!("Model loaded, starting inference test...");

// Lepas loop untuk process directory
// Ubah ke:
// let test_video = "../samples/video_asakai.mp4";
// video_processor.process_video(Path::new(test_video))?;
```

Rebuild dan run:
```powershell
cargo build --release
cargo run --release
```

## Performance Considerations

### RAM Usage:
- Model loading: ~500MB
- Per frame inference: ~100-300MB
- Total: ~1-2 GB recommended

### GPU (Optional):

ONNX Runtime dapat use GPU (CUDA/TensorRT) untuk faster inference:
```toml
# Update Cargo.toml (experimental):
ort = { version = "1.16", features = ["cuda", "download-binaries"] }
```

Requires: CUDA Toolkit installed

### CPU Optimization:

Automatic via ONNX Runtime optimization levels:
- Default: All available optimizations
- Custom: Set di detector initialization

## Next Steps

1. ✅ Install prerequisites
2. ✅ Build project
3. ✅ Prepare model dan videos
4. ✅ Run detector
5. ✅ Check output videos

## Support

For issues:
1. Check OpenCV installation: `$env:OpenCV_DIR`
2. Check ONNX Runtime: Run dengan verbose
3. Check model path di `src/main.rs`
4. Verify video formats

## Additional Resources

- Rust Book: https://doc.rust-lang.org/book/
- OpenCV Rust: https://docs.rs/opencv/
- ONNX Runtime: https://github.com/microsoft/onnxruntime
- YOLOv8 Format: https://docs.ultralytics.com/
