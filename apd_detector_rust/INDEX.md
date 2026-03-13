# APD Detector Rust - Complete Project

## 📋 Project Structure

```
apd_detector_rust/
│
├── 📄 Cargo.toml                      # Project manifest & dependencies
├── 📄 .gitignore                      # Git ignore rules
│
├── src/                               # Source code
│   ├── main.rs                        # Entry point
│   ├── detector.rs                    # ONNX model inference (★ Core)
│   ├── processor.rs                   # Video processing pipeline (★ Core)
│   ├── utils.rs                       # Utilities: Detection, NMS, drawing
│   └── error.rs                       # Custom error types
│
├── examples/
│   └── advanced.rs                    # Advanced usage examples
│
├── 🔧 run.bat                         # Windows runner script
├── 🔧 run.sh                          # Linux/Mac runner script
│
└── 📚 Documentation/
    ├── QUICKSTART.md                  # ★ START HERE (5 min)
    ├── SETUP.md                       # Detailed installation
    ├── README.md                      # Feature overview
    ├── DOCS.md                        # Complete documentation
    ├── CONFIG.md                      # Configuration guide
    └── INDEX.md                       # This file
```

---

## 🚀 Getting Started (Choose Your Path)

### ⏱️ Path 1: Want to Run ASAP? (5 minutes)
1. Read: [QUICKSTART.md](QUICKSTART.md)
2. Run: `cargo run --release`
3. Check output in: `../output_videos/`

**Best for:** Impatient people or those with Rust + OpenCV already installed

### 📖 Path 2: Want Complete Setup? (30 minutes)
1. Read: [SETUP.md](SETUP.md) - Step by step installation
2. Build: `cargo build --release`
3. Run: `cargo run --release` or `./run.bat`

**Best for:** First time setup or new machine

### 🔧 Path 3: Want to Customize? (45 minutes)
1. Read: [SETUP.md](SETUP.md) - Install dependencies
2. Read: [DOCS.md](DOCS.md) - Understand architecture
3. Read: [CONFIG.md](CONFIG.md) - Change settings
4. Edit: `src/main.rs`, `src/detector.rs`
5. Build & test

**Best for:** Developers customizing thresholds or adding features

### 📚 Path 4: Want to Understand Everything? (2 hours)
1. [QUICKSTART.md](QUICKSTART.md) - Quick overview
2. [SETUP.md](SETUP.md) - Installation details
3. [README.md](README.md) - Features & usage
4. [DOCS.md](DOCS.md) - Architecture & implementation
5. [CONFIG.md](CONFIG.md) - All configuration options
6. Read source code: `src/` directory

**Best for:** Those who like to understand how things work

---

## 📚 Documentation Map

| Document | Duration | Purpose |
|----------|----------|---------|
| [QUICKSTART.md](QUICKSTART.md) | 5 min | Get running ASAP |
| [SETUP.md](SETUP.md) | 30 min | Install everything |
| [README.md](README.md) | 10 min | Features & overview |
| [DOCS.md](DOCS.md) | 30 min | Deep dive architecture |
| [CONFIG.md](CONFIG.md) | 15 min | Tuning thresholds |
| **Source code** | varies | Implementation details |

---

## 🎯 Common Tasks

### ✅ I want to process videos
```bash
cargo run --release
```
→ See [QUICKSTART.md](QUICKSTART.md)

### ✅ I need to install Rust
```powershell
irm https://sh.rustup.rs | iex
```
→ See [SETUP.md - Prerequisites](SETUP.md#prerequisites)

### ✅ I need to install OpenCV
Try your OS:
- **Windows**: [SETUP.md - OpenCV (Windows)](SETUP.md#3-install-opencv)
- **Mac**: `brew install opencv`
- **Linux**: `sudo apt-get install libopencv-dev`

### ✅ I want to change confidence threshold
Edit `src/detector.rs`:
```rust
conf_threshold: 0.5,  // Change this value
```
→ See [CONFIG.md - Confidence & IoU Threshold](CONFIG.md#confidence--iou-threshold)

### ✅ I want more/fewer detections
**More detections** (lower threshold):
```rust
detector.set_conf_threshold(0.3);
```

**Fewer detections** (higher threshold):
```rust
detector.set_conf_threshold(0.7);
```

### ✅ I need to use different model
Edit `src/main.rs`:
```rust
let model_path = "../path/to/your/model.onnx";
let classes = vec!["your", "classes", "here"];
```

### ✅ Build takes too long
First build is slow (ONNX downloads ~5-10 min). Subsequent builds are faster:
```bash
cargo build --release  # First: ~20 min
cargo build --release  # Next: ~1-5 min
```

### ✅ I got build errors
Most common:
1. OpenCV not found → [SETUP.md - Troubleshooting](SETUP.md#troubleshooting-build)
2. MSVC not installed → [SETUP.md - MSVC](SETUP.md#2-install-msvc-build-tools-required)
3. First time ONNX → Wait & be patient, it's downloading (~200MB)

---

## 🔍 Source Code Overview

| File | Lines | Purpose |
|------|-------|---------|
| `main.rs` | ~40 | Entry point, configuration |
| `detector.rs` | ~280 | ★ Core: ONNX inference |
| `processor.rs` | ~200 | ★ Core: Video processing |
| `utils.rs` | ~250 | Detection, NMS, drawing |
| `error.rs` | ~30 | Error handling |

**★ Most important**: `detector.rs` and `processor.rs`

---

## 📊 Model Information

- **Model**: YOLOv26
- **Format**: ONNX
- **Location**: `../model/yolov26_retrain/best.onnx`
- **Input**: 640×640 RGB images (normalized 0-1)
- **Output**: Detections with bounding boxes

### Classes (3 total)
```
0 = helmet (helm)                - Green box
1 = person (orang)               - Blue box
2 = fire-extinguisher (pemadam)  - Red box
```

---

## 🎮 Usage Quick Reference

```bash
# Build & run (release - faster)
cargo run --release

# Just build
cargo build --release

# Build debug (for development)
cargo build

# Run debug build
cargo run

# Run advanced examples
cargo run --example advanced --release

# Clean build artifacts
cargo clean

# Windows batch scripts
.\run.bat              # Release mode
.\run.bat debug        # Debug mode
.\run.bat clean        # Clean
.\run.bat help         # Help

# Linux/Mac scripts
./run.sh               # Release mode
./run.sh debug         # Debug mode
./run.sh clean         # Clean
./run.sh help          # Help
```

---

## ⚙️ Configuration Checklist

Before running, verify:

- [x] Model exists: `../model/yolov26_retrain/best.onnx`
- [x] Videos exist: `../samples/*.mp4` (or other video format)
- [x] Output dir: `../output_videos/` (auto-created)
- [x] Confidence threshold: Set in `src/detector.rs` (default: 0.5)
- [x] IoU threshold: Set in `src/detector.rs` (default: 0.45)
- [x] Classes match model: `src/main.rs` (should be 3 classes)

---

## 📈 Performance Notes

| Resolution | Speed | GPU | CPU |
|------------|-------|-----|-----|
| 640×480 | ~30-60 FPS | RTX 3070 | 5-10 FPS |
| 1080p | ~15-30 FPS | RTX 3070 | 2-5 FPS |
| 4K | ~5-10 FPS | RTX 3070 | <1 FPS |

Performance depends on:
- GPU/CPU speed
- Number of objects
- Model complexity
- Video codec

---

## 🆘 Troubleshooting Guide

### Build Fails
→ [SETUP.md - Troubleshooting Build](SETUP.md#troubleshooting-build)

### Runtime Errors
→ [DOCS.md - Troubleshooting](DOCS.md#troubleshooting)

### Slow Performance
→ [CONFIG.md - Performance Options](CONFIG.md#performance-options)

### Video Not Processing
→ Check video format, file permissions, disk space

---

## 🎓 Learning Path

For those new to Rust:
1. [QUICKSTART.md](QUICKSTART.md) - Get it running
2. [src/main.rs](src/main.rs) - Understand flow
3. [src/detector.rs](src/detector.rs) - Learn inference
4. [src/processor.rs](src/processor.rs) - Learn video processing
5. [DOCS.md](DOCS.md) - Deep dive

---

## 📝 Key Concepts

### ONNX Runtime
- 🎯 Why? Fast, lightweight, cross-platform
- 🚀 No Python overhead like Ultralytics
- 📦 Includes optimizations (quantization, graph fusion, etc.)

### NMS (Non-Maximum Suppression)
- 🎯 Why? Remove overlapping detections
- 📊 Threshold: 0.45 (tune in `src/detector.rs`)
- ⚙️ Applied after all detections are collected

### Preprocessing
- 📐 Resize: 640×640 (model input size)
- 🎨 BGR→RGB: OpenCV uses BGR by default
- 📊 Normalize: 0-255 → 0-1 for ONNX
- 🔄 Channel-first: PyTorch format (C, H, W)

### Bounding Boxes
- 📍 Input: Model outputs as (cx, cy, w, h)
- 📍 Output: Converted to (x1, y1, x2, y2) for drawing
- 📏 Scaled: From 640×640 model space to original video size

---

## 💡 Tips & Tricks

1. **Faster iteration**: Use debug builds, slower but compiles quicker
2. **Memory usage**: Frame-by-frame processing keeps memory low
3. **Output video**: Save as H.264 MP4 for compatibility
4. **Label accuracy**: Labels preserved exactly from model
5. **Batch processing**: Process multiple videos sequentially

---

## 🔗 Links

- **Rust**: https://www.rust-lang.org/
- **ONNX Runtime**: https://onnxruntime.ai/
- **OpenCV**: https://opencv.org/
- **YOLOv8**: https://docs.ultralytics.com/
- **ONNX**: https://onnx.ai/

---

## 📞 Support

For issues:
1. Check relevant documentation file
2. Review troubleshooting section
3. Check source code comments
4. Review error messages carefully

---

## 📋 Checklist Before Using

- [ ] Rust installed? (`rustc --version`)
- [ ] OpenCV installed? ($env:OpenCV_DIR)
- [ ] Model file exists? (`model/yolov26_retrain/best.onnx`)
- [ ] Videos prepared? (`samples/*.mp4`)
- [ ] Read QUICKSTART.md? ✓
- [ ] Ready to run? ✓

---

**Version**: 1.0.0-beta  
**Updated**: 2026-03-12  
**Status**: ✅ Ready to use

**Next Step**: Start with [QUICKSTART.md](QUICKSTART.md) or [SETUP.md](SETUP.md)
