# Quick Start - APD Detector Rust

## TL;DR - Langsung Jalan

### Sudah ada Rust + OpenCV?

```bash
cd apd_detector_rust
cargo run --release
```

Output akan tersimpan di `../output_videos/`

---

## 5 Menit Setup

### 1. Install Rust (5 menit)

**Windows:**
```powershell
irm https://sh.rustup.rs | iex
```

**Mac/Linux:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Install OpenCV (10 menit)

**Windows (vcpkg):**
```powershell
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install opencv:x64-windows

# Add ke environment
[Environment]::SetEnvironmentVariable("OpenCV_DIR", "C:\...\vcpkg\installed\x64-windows", "User")
```

**Mac:**
```bash
brew install opencv
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libopencv-dev
```

### 3. Build & Run (2 menit)

```bash
cd apd_detector_rust
cargo build --release
cargo run --release
```

**Done!** Cek output di `output_videos/`

---

## Troubleshooting

### ❌ "OpenCV not found"
```powershell
$env:OpenCV_DIR = "C:\path\to\opencv"
cargo clean
cargo build --release
```

### ❌ "ort crate fails"
```bash
# Butuh internet connection, tunggu sampai selesai
cargo build --release --verbose
```

### ❌ "MSVC compiler not found"
Install: https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## Customization

### Change Confidence Threshold

Edit `src/detector.rs`:
```rust
conf_threshold: 0.5,  // 0.0-1.0 (lower = lebih banyak detections)
```

### Change Classes

Edit `src/main.rs`:
```rust
let classes = vec!["helm", "person", "pemadam"];  // Sesuaikan nama
```

### Change Model Path

Edit `src/main.rs`:
```rust
let model_path = "../model/yolabc.onnx";  // Path ke model ONNX
```

---

## Running Options

### Process semua video di folder:
```bash
cargo run --release
```

### Debug mode (for development):
```bash
cargo build
cargo run
```

### Batch script (Windows):
```powershell
.\run.bat           # Release mode
.\run.bat debug     # Debug mode
.\run.bat clean     # Clean build files
```

### Bash script (Linux/Mac):
```bash
./run.sh            # Release mode
./run.sh debug      # Debug mode
./run.sh clean      # Clean build files
```

---

## File Locations

```
c:\Magang_Alif\Trial\
├── model/
│   └── yolov26_retrain/
│       └── best.onnx              ← Model ONNX
├── samples/
│   ├── kaizen_casting.mp4         ← Input video
│   ├── workshop_kaizen.mp4
│   └── ...
├── output_videos/                 ← Output (created automatically)
│   ├── detected_kaizen_casting.mp4
│   └── ...
└── apd_detector_rust/
    └── (Rust project files)
```

---

## Performance

Estimasi kecepatan processing:

| Hardware | Resolution | Speed |
|----------|------------|-------|
| RTX 3070 | 640x480    | 30-60 FPS |
| RTX 3070 | 1080p      | 15-30 FPS |
| CPU only | 640x480    | 5-10 FPS |
| CPU only | 1080p      | 2-5 FPS |

---

## Advanced

### Run advanced examples:
```bash
cargo run --example advanced --release
```

### Custom single frame detection:
```bash
# Edit source, uncomment example_single_frame()
cargo run --example advanced --release
```

### Batch processing dengan stats:
```bash
# Edit source, uncomment example_statistics()
cargo run --example advanced --release
```

---

## Model Classes

- **0: helmet** (Helm, warna hijau)
- **1: person** (Orang, warna biru)
- **2: fire-extinguisher** (Pemadam, warna merah)

Bounding box label TIDAK akan kacau - sesuai dengan model ONNX output.

---

## Logs & Monitoring

Output JSON detections (optional):

Edit `src/processor.rs`, uncomment:
```rust
save_detections_json(&detections, "detections.json")?;
```

---

## Help

Dokumentasi lengkap:
- `README.md` - Feature details
- `SETUP.md` - Detailed setup
- `CONFIG.md` - Configuration options
- `examples/advanced.rs` - Code examples

---

## Next Steps

1. ✅ Install prerequisites
2. ✅ Build project
3. ✅ Run detector
4. 🔄 Monitor output
5. 📊 Analyze results

Selamat mencoba! 🚀
