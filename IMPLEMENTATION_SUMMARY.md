# APD Detector Rust - Complete Implementation

## ✅ What Has Been Created

Berikut adalah file Rust yang sudah dibuat untuk deteksi APD dengan ONNX Runtime.

### Project Root (`apd_detector_rust/`)

```
apd_detector_rust/
├── Cargo.toml                    ✅ Project configuration & dependencies
├── .gitignore                    ✅ Git ignore patterns
├── run.bat                       ✅ Windows runner script
├── run.sh                        ✅ Linux/Mac runner script
└── src/
    ├── main.rs                   ✅ Entry point & configuration
    ├── detector.rs               ✅ ONNX inference engine (★ CORE)
    ├── processor.rs              ✅ Video processing pipeline (★ CORE)
    ├── utils.rs                  ✅ Detection utilities & drawing
    └── error.rs                  ✅ Custom error types
```

### Documentation

```
├── INDEX.md                      ✅ Navigation guide (START HERE)
├── QUICKSTART.md                 ✅ 5-minute quick start
├── SETUP.md                      ✅ Detailed installation guide
├── README.md                     ✅ Features & overview
├── DOCS.md                       ✅ Complete technical documentation
└── CONFIG.md                     ✅ Configuration & tuning guide
```

### Examples

```
└── examples/
    └── advanced.rs               ✅ Advanced usage examples
```

---

## 🎯 What Each File Does

### Core Files (Run the detector)

**`src/main.rs`**
- Entry point
- Configures model path & classes
- Sets input/output directories
- Creates detector and processor
- Starts batch video processing
- Customizable: Change paths, classes, thresholds

**`src/detector.rs` ⭐**
- Loads ONNX model via ort runtime
- Manages inference session
- Preprocesses frames (resize, normalize, channel-first)
- Runs model inference
- Parses YOLOv26 output (8400×84 format)
- Applies confidence filtering
- Returns Vec<Detection>

**`src/processor.rs` ⭐**
- Opens video files
- Extracts frame properties (resolution, FPS, total frames)
- For each frame:
  - Preprocesses with detector
  - Runs inference
  - Draws bounding boxes
  - Writes to output video
- Handles video codec automatically
- Shows progress reporting

**`src/utils.rs`**
- Detection struct with bounding box coordinates
- IOU calculation (for NMS)
- NMS implementation (removes overlapping detections)
- ColorPalette for class-specific colors
- draw_detections() function
- preprocess_frame() (resize + normalize)
- save_detections_json() for export

**`src/error.rs`**
- Custom error types
- Better error messages
- Type-safe error handling

### Configuration Files

**`Cargo.toml`**
- Rust project manifest
- Dependencies:
  - `ort` 1.16 - ONNX Runtime
  - `opencv` 0.91 - Video/image processing
  - `ndarray` - Tensor operations
  - `tokio` - Async
  - `serde` - JSON serialization

**`.gitignore`**
- Ignores build artifacts
- Ignores IDE files
- Ignores output videos

### Runner Scripts

**`run.bat` (Windows)**
```
Usage: run.bat [debug|release|clean|help]
- Checks Rust & model
- Builds & runs
- Error handling
```

**`run.sh` (Linux/Mac)**
```
Same functionality as run.bat
```

### Documentation Files

**`INDEX.md`** ← **Start here**
- Navigation guide
- 4 different getting-started paths
- Quick reference
- Troubleshooting links

**`QUICKSTART.md`**
- 5-minute setup
- TL;DR commands
- Common customizations

**`SETUP.md`**
- Step-by-step installation
- Windows/Mac/Linux instructions
- OpenCV setup (vcpkg, brew, apt)
- Troubleshooting build issues

**`README.md`**
- Features overview
- Project structure
- Usage examples
- Performance notes
- Configuration options

**`DOCS.md`**
- Complete technical documentation
- Architecture diagrams
- Data flow explanation
- Module details
- Model output format
- Performance tuning
- Advanced usage
- Known limitations

**`CONFIG.md`**
- Configuration reference
- Threshold tuning
- Performance options
- Advanced tuning recommendations

**`examples/advanced.rs`**
- Single frame detection
- Custom thresholds
- Batch processing with progress
- Filter detections by class
- Statistics calculation
- Performance benchmarking

---

## 🚀 How to Use

### Step 1: Install Prerequisites (if needed)
See [SETUP.md](apd_detector_rust/SETUP.md)

### Step 2: Navigate to Project
```bash
cd c:\Magang_Alif\Trial\apd_detector_rust
```

### Step 3: Build & Run
```bash
cargo run --release
```

### Step 4: Check Output
```
c:\Magang_Alif\Trial\output_videos\
```

---

## 🔧 Configuration

### Change Detection Classes
Edit `src/main.rs`:
```rust
let classes = vec!["helmet", "person", "fire-extinguisher"];
```

### Change Model Path
Edit `src/main.rs`:
```rust
let model_path = "../model/yolov26_retrain/best.onnx";
```

### Change Input/Output Directories
Edit `src/main.rs`:
```rust
let video_dir = "../samples";
let output_dir = "../output_videos";
```

### Change Confidence Threshold
Edit `src/detector.rs`:
```rust
conf_threshold: 0.5,  // 0.0 = all detections, 1.0 = none
```

### Change NMS IoU Threshold
Edit `src/detector.rs`:
```rust
iou_threshold: 0.45,  // Lower = stricter removal of overlaps
```

---

## 📊 Model Details

- **Type**: YOLOv26 (object detection)
- **Format**: ONNX (Open Neural Network Exchange)
- **Location**: `../model/yolov26_retrain/best.onnx`
- **Input Size**: 640×640 normalized RGB images
- **Classes**: 3 (helmet, person, fire-extinguisher)
- **Output**: Detections with bounding boxes

### How It Works

```
1. Frame from video (e.g. 1080×1920)
   ↓
2. Resize to 640×640
   ↓
3. Normalize pixels (0-255 → 0-1)
   ↓
4. From BGR to RGB color space
   ↓
5. Convert to channel-first format
   ↓
6. Run ONNX inference
   ↓
7. Parse output detections
   ↓
8. Apply NMS (remove overlaps)
   ↓
9. Scale boxes back to original size
   ↓
10. Draw boxes on frame
   ↓
11. Write to output video
```

---

## 🎯 Key Features Implemented

✅ **ONNX Runtime** (NOT Ultralytics)
- Fast, lightweight, cross-platform
- No Python overhead
- Built-in optimizations

✅ **Video Processing**
- Batch video processing
- Automatic codec detection
- Frame-by-frame streaming
- Progress reporting

✅ **Detection**
- Multi-class support (3 classes)
- Confidence filtering
- Non-Maximum Suppression
- Bounding box scaling

✅ **Bounding Boxes**
- Preserved labels (not messed up)
- Class-specific colors
- Proper scaling to original resolution

✅ **Error Handling**
- Graceful error recovery
- Detailed error messages
- File/stream handling

✅ **Performance**
- Multi-threaded inference
- ONNX optimization levels
- Efficient memory usage

---

## 📈 Performance Metrics (Expected)

| Resolution | FPS (GPU) | FPS (CPU) |
|-----------|-----------|-----------|
| 640×480 | 30-60 | 5-10 |
| 1080p | 15-30 | 2-5 |
| 4K | 5-10 | <1 |

*Assumes RTX 3070 or similar for "GPU"*

---

## ✨ Special Features

### ✅ Labels Not Messed Up
Bounding box labels are preserved exactly as the model outputs them. No confusion or overlapping text.

### ✅ Color Coding
- **Helmet** = Green boxes
- **Person** = Blue boxes
- **Fire-Extinguisher** = Red boxes

### ✅ Batch Processing
Automatically processes all video files (.mp4, .avi, .mov, .mkv, .flv, .wmv) in a directory.

### ✅ Progress Tracking
Shows real-time progress:
- Current frame / total frames
- Processing speed
- Estimated time remaining

---

## 🆘 Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| "OpenCV not found" | [SETUP.md - OpenCV Installation](apd_detector_rust/SETUP.md#3-install-opencv) |
| "ONNX build fails" | [SETUP.md - ONNX Runtime](apd_detector_rust/SETUP.md#troubleshooting-build) |
| "Rust not installed" | [SETUP.md - Prerequisites](apd_detector_rust/SETUP.md#prerequisites) |
| "Model not found" | Check path in `src/main.rs` |
| "Slow performance" | [CONFIG.md - Performance](apd_detector_rust/CONFIG.md#performance-options) |
| "Too many detections" | Increase `conf_threshold` in `src/detector.rs` |
| "Missing detections" | Decrease `conf_threshold` in `src/detector.rs` |

---

## 📚 Where to Start

### 👤 I just want to run it
→ [QUICKSTART.md](apd_detector_rust/QUICKSTART.md)

### 🏗️ I need to install everything
→ [SETUP.md](apd_detector_rust/SETUP.md)

### 📖 I want to understand how it works
→ [DOCS.md](apd_detector_rust/DOCS.md)

### 🎛️ I want to adjust settings
→ [CONFIG.md](apd_detector_rust/CONFIG.md)

### 🗺️ I'm lost
→ [INDEX.md](apd_detector_rust/INDEX.md) - Navigation guide

---

## 🎓 Code Quality

- ✅ Error handling via `Result<T>` & custom error types
- ✅ Type-safe with Rust's ownership system
- ✅ No memory leaks or data races
- ✅ Efficient tensor operations with `ndarray`
- ✅ Clean separation of concerns (detector, processor, utils)
- ✅ Well-commented code
- ✅ Async-ready with `tokio`

---

## 🔄 Workflow

```
1. User runs: cargo run --release
   ↓
2. main.rs creates detector & processor
   ↓
3. processor finds all video files in samples/
   ↓
4. For each video:
   a. VideoCapture opens the file
   b. Read each frame
   c. detector.detect() on each frame
   d. draw_detections() adds boxes
   e. VideoWriter saves to output
   ↓
5. Output videos saved to output_videos/
   ↓
6. Done!
```

---

## 📦 Deliverables

| Item | Status | Location |
|------|--------|----------|
| Source code | ✅ | `apd_detector_rust/src/` |
| Documentation | ✅ | `apd_detector_rust/` |
| Build scripts | ✅ | `apd_detector_rust/{run.bat, run.sh}` |
| Examples | ✅ | `apd_detector_rust/examples/` |
| Configuration | ✅ | Built-in via `src/` |

---

## 🎉 You're Ready!

Everything is set up. Just:

```bash
cd apd_detector_rust
cargo run --release
```

Output will be in: `../output_videos/`

---

**Questions?** Check [INDEX.md](apd_detector_rust/INDEX.md) for navigation to relevant documentation.

**Version**: 1.0.0-beta  
**Created**: 2026-03-12  
**Status**: ✅ Complete & Ready to Use
