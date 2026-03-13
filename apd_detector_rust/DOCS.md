# APD Detector - Dokumentasi Lengkap

## Overview

APD Detector adalah aplikasi Rust untuk mendeteksi Personal Protective Equipment (PPE) menggunakan:
- **Model**: YOLOv26 (format ONNX)
- **Runtime**: ONNX Runtime (bukan Ultralytics)
- **Input**: Video dari folder `samples/`
- **Output**: Video dengan bounding boxes

## Fitur Utama

### ✅ Implemented
- [x] ONNX Runtime inference (high performance, lightweight)
- [x] YOLOv26 model loading
- [x] Multi-format video support (mp4, avi, mov, mkv, flv, wmv)
- [x] Batch video processing
- [x] Non-Maximum Suppression (NMS) untuk overlapping detections
- [x] Bounding box drawing dengan preserved labels
- [x] Progress reporting per-frame
- [x] Automatic output directory creation
- [x] Error handling dan recovery

### 🚀 Performance
- [x] Multi-threaded inference
- [x] ONNX optimization levels
- [x] Frame-by-frame streaming (low memory)
- [x] GPU support ready (via ONNX)

### 📝 Logging & Debugging
- [x] Console progress output
- [x] Frame detection statistics
- [x] Error messages terstruktur
- [x] Performance metrics

## Architecture

```
apd_detector_rust/
├── src/
│   ├── main.rs           # Entry point
│   ├── detector.rs       # ONNX model inference
│   ├── processor.rs      # Video processing pipeline
│   ├── utils.rs          # Detection, NMS, drawing
│   └── error.rs          # Custom error types
├── examples/
│   └── advanced.rs       # Advanced usage examples
├── Cargo.toml            # Dependencies
├── README.md             # Feature overview
├── SETUP.md              # Installation guide
├── QUICKSTART.md         # Quick start guide
├── CONFIG.md             # Configuration
└── run.bat/run.sh        # Runner scripts
```

## Module Details

### `detector.rs` - Core Inference

**APDDetector struct:**
- Loads ONNX model
- Manages inference session
- Parses YOLOv8 format output
- Handles confidence/IoU thresholds

**Key methods:**
```rust
pub fn new(model_path: &str, classes: &[&str]) -> Result<Self>
pub fn detect(&self, frame_data: &[f32], frame_height: usize, frame_width: usize) -> Result<Vec<Detection>>
pub fn set_conf_threshold(&mut self, threshold: f32)
pub fn set_iou_threshold(&mut self, threshold: f32)
```

**Output format**: Detection struct dengan fields:
- `x1, y1, x2, y2` - bounding box coordinates
- `confidence` - detection confidence (0-1)
- `class_id` - class index (0-2)
- `class_name` - class label string

### `processor.rs` - Video Pipeline

**VideoProcessor struct:**
- Opens video files
- Reads frames sequentially
- Preprocesses frames
- Writes output video
- Handles errors gracefully

**Key methods:**
```rust
pub fn process_videos_in_directory(&self, video_dir: &str) -> Result<()>
pub fn process_video(&self, video_path: &Path) -> Result<()>
pub fn process_frame(&self, frame: &Mat) -> Result<()>
```

**Processing flow:**
1. Open video (CAP_ANY codec detection)
2. Extract properties (resolution, FPS, total frames)
3. For each frame:
   - Read frame
   - Preprocess (resize + normalize)
   - Run inference
   - Draw detections
   - Write to output
4. Save output video

### `utils.rs` - Detection & Drawing

**Detection struct:**
```rust
pub struct Detection {
    pub x1: f32, pub y1: f32,
    pub x2: f32, pub y2: f32,
    pub confidence: f32,
    pub class_id: usize,
    pub class_name: String,
}
```

**Functions:**
- `Detection::iou()` - Intersection over Union
- `NonMaxSuppression::apply()` - Remove overlapping boxes
- `ColorPalette::get_color()` - Class-specific colors
- `draw_detections()` - Draw boxes and labels
- `preprocess_frame()` - Normalize for model
- `save_detections_json()` - Export to JSON

### `error.rs` - Error Handling

Custom error types untuk clear error messages:
- `ModelLoadError` - Model fail to load
- `InferenceError` - Runtime errors
- `VideoProcessingError` - Video I/O issues
- `OnnxRuntimeError` - ONNX runtime issues
- `OpenCVError` - OpenCV issues

## Data Flow

```
Video File
    ↓
[VideoCapture]  (OpenCV)
    ↓
Frame (Mat)
    ↓
[Preprocessing]
    ├─ Resize to 640x640
    ├─ BGR→RGB convert
    ├─ Normalize 0-255 → 0-1
    └─ Channel-first format
    ↓
Normalized tensor (1, 3, 640, 640)
    ↓
[ONNX Inference]
    ├─ Forward pass
    └─ Get outputs
    ↓
Raw output (1, 8400, 84)
    ├─ Format: [cx, cy, w, h, conf, class_probs...]
    └─ In model's 640x640 space
    ↓
[Parse & Scale]
    ├─ Filter by confidence
    ├─ Get max class confidence
    └─ Scale to original resolution
    ↓
Detections (Vec<Detection>)
    ↓
[NMS]
    └─ Remove overlaps (IOU > 0.45)
    ↓
Final detections
    ↓
[Draw & Output]
    ├─ Draw boxes (BGR colored)
    ├─ Draw labels
    └─ Write frame
    ↓
Output Video File
```

## Model Output Format

YOLOv26 ONNX output:
```
Shape: (batch, num_detections, 84)
       (1,     8400,             84)

Per detection (84 values):
├─ Index 0-3: bbox center [cx, cy, w, h] (in 640x640 space)
├─ Index 4: objectness confidence
└─ Index 5-84: class probabilities (3 classes)
  ├─ Index 5: helmet
  ├─ Index 6: person
  └─ Index 7: fire-extinguisher
```

## Configuration Options

### Model Setup
```rust
// File: src/main.rs
let model_path = "../model/yolov26_retrain/best.onnx";
let classes = vec!["helmet", "person", "fire-extinguisher"];
```

### Thresholds
```rust
// File: src/detector.rs
conf_threshold: 0.5,    // Min objectness
iou_threshold: 0.45,    // NMS threshold
```

### Input/Output
```rust
let video_dir = "../samples";           // Input videos
let output_dir = "../output_videos";    // Output videos
```

## Performance Tuning

### For More Detections (Lower Thresholds)
```rust
detector.set_conf_threshold(0.3);  // Accept low confidence
detector.set_iou_threshold(0.3);   // Stricter NMS (fewer removed)
```

### For Fewer False Positives (Higher Thresholds)
```rust
detector.set_conf_threshold(0.7);  // Accept only high confidence
detector.set_iou_threshold(0.5);   // Looser NMS (more removed)
```

### By Resolution
```
Video: 640x480  → Process at native size
Video: 1080p    → Preprocess handles resize
Video: 4K       → Slower but works
```

## Compilation Details

### Dependencies

**ONNX Runtime:**
- Version: 1.16+
- Auto-downloads platform-specific binaries
- First time: 5-10 minutes + 200MB download

**OpenCV:**
- Version: 0.91 (Rust bindings)
- Requires system OpenCV development files
- Windows: vcpkg recommended
- Mac/Linux: package managers

**Other:**
- `ndarray` - Tensor operations
- `tokio` - Async support
- `serde` - Serialization

### Build Times

| Build Type | Time | Size |
|-----------|------|------|
| First debug | 15-20m | 800MB |
| First release | 20-30m | 100MB |
| Incremental debug | 30s-2m | - |
| Incremental release | 1-5m | - |

## Advanced Usage

### Single Frame Detection
```rust
// Load image
let frame = imgcodecs::imread("image.jpg", IMREAD_COLOR)?;

// Create detector
let detector = APDDetector::new(model_path, &classes)?;

// Get processor
let processor = VideoProcessor::new(&detector, output_dir);

// Process
processor.process_frame(&frame)?;
```

### Custom Batch Processing
```rust
let mut detector = APDDetector::new(model_path, &classes)?;
detector.set_conf_threshold(0.7);
detector.set_iou_threshold(0.4);

// ... process videos
```

### Filter Detections
```rust
let filtered: Vec<_> = detections
    .into_iter()
    .filter(|d| d.class_name == "helmet")
    .collect();
```

## Troubleshooting

### Model Issues
- ✅ Model path correct?
- ✅ File exists and readable?
- ✅ ONNX format correct?

### Video Issues
- ✅ Format supported (mp4, avi, mov, mkv)?
- ✅ Codec supported by system?
- ✅ File not corrupted?

### Performance
- ✅ GPU available?
- ✅ RAM sufficient (~2GB)?
- ⚠️ Disk space for output?

### Build Issues
- ✅ Rust updated? `rustup update`
- ✅ MSVC installed?
- ✅ OpenCV libs found? `$env:OpenCV_DIR`

## Known Limitations

- ⚠️ Single model only (hardcoded class count)
- ⚠️ No real-time camera support (video files only)
- ⚠️ No GPU CUDA support (default ONNX CPU)
- ⚠️ No configuration file (hardcoding in source)

## Future Improvements

- [ ] YAML config file support
- [ ] REST API server
- [ ] Real-time camera input
- [ ] Multi-model support
- [ ] GPU acceleration (CUDA/TensorRT)
- [ ] Web UI dashboard
- [ ] Streaming output
- [ ] Database logging

## References

- **ONNX Runtime**: https://onnxruntime.ai/
- **Rust OpenCV**: https://docs.rs/opencv/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Model Format**: MS ONNX specification

## Support & Contributing

- Issues: Report dengan model version + video sample
- Improvements: Suggestions welcome
- Performance: Benchmark results appreciated

## License

Internal use only.

---

**Last Updated**: 2026-03-12
**Version**: 1.0.0-beta
**Rust Edition**: 2021
