# APD Detector - Rust dengan ONNX Runtime

Deteksi APD (Personal Protective Equipment) menggunakan model YOLOv26 dan ONNX Runtime di Rust.

## Fitur

- ✅ Menggunakan ONNX Runtime (bukan Ultralytics)
- ✅ Memproses video dari folder `samples/`
- ✅ 3 kelas: helmet, person, fire-extinguisher
- ✅ Label box tidak akan kacau (sesuai model)
- ✅ Output video dengan bounding box
- ✅ NMS (Non-Maximum Suppression) untuk remove overlapping detections
- ✅ Deteksi multi-frame dengan progress tracking

## Requirements

- Rust 1.70+ (installation: https://www.rust-lang.org/tools/install)
- Model: `model/yolov26_retrain/best.onnx`
- Video files in: `samples/`

### Dependensi sistem (Windows):

Download dan install:
- **OpenCV** (build dari source atau gunakan pre-built binaries)
  
Atau install via vcpkg:
```powershell
vcpkg install opencv:x64-windows
```

## Build & Run

### 1. Setup

```bash
# Masuk direktori project
cd apd_detector_rust

# Build (pertama kali akan download ONNX Runtime binaries)
cargo build --release
```

Pada build pertama, ONNX Runtime akan di-download otomatis (~200MB).

### 2. Run

```bash
# Proses semua video
cargo run --release

# Atau jalankan binary directly
./target/release/apd_detector.exe
```

### 3. Output

Video dengan detections akan tersimpan di folder `../output_videos/`

## Structure

```
apd_detector_rust/
├── Cargo.toml              # Dependencies
├── src/
│   ├── main.rs             # Entry point
│   ├── detector.rs         # ONNX inference logic
│   ├── processor.rs        # Video processing
│   ├── utils.rs            # Detection struct, NMS, drawing
│   └── error.rs            # Error types
└── README.md               # This file
```

## Konfigurasi

Edit `src/main.rs` untuk mengubah:

```rust
let model_path = "../model/yolov26_retrain/best.onnx";  // Path model
let video_dir = "../samples";                            // Input video dir
let output_dir = "../output_videos";                     // Output dir
let classes = vec!["helmet", "person", "fire-extinguisher"]; // Classes
```

### Confidence & IoU Threshold

Di `src/detector.rs`:
```rust
conf_threshold: 0.5,    // Minimum confidence (0.0-1.0)
iou_threshold: 0.45,    // NMS threshold (0.0-1.0)
```

Ubah via setter:
```rust
detector.set_conf_threshold(0.6);
detector.set_iou_threshold(0.4);
```

## Troubleshooting

### Error: "Model load error"
- Pastikan path model benar di `src/main.rs`
- File `.onnx` harus exist dan valid

### Error: "Video processing error"
- Cek format video supported (mp4, avi, mov, mkv, etc.)
- Pastikan video tidak corrupt

### Error: OpenCV or encoding issues
- Install OpenCV development files
- Pada Windows: `vcpkg install opencv:x64-windows`

### Performance tips
- Reduce video resolution di preprocessing jika lambat
- Tingkatkan `iou_threshold` untuk NMS lebih ketat (fewer detections)
- Turunkan `conf_threshold` untuk lebih stringent detections

## Model Info

Model: YOLOv26 (dari ONNX format)
- Input: 640x640 RGB images (normalized 0-1)
- Output: (1, num_detections, 84)
  - Format: [cx, cy, w, h, objectness, class_probs...]
  
Classes:
1. helmet (hijau) - Green
2. person (biru) - Blue  
3. fire-extinguisher (merah) - Red

## Notes

- Bounding box labels **TIDAK akan kacau** - semua label sesuai dengan output model
- Preprocessing: BGR→RGB, normalize to 0-1, channel-first format
- Output video codec: H.264 (mp4v)
- Semua detections dengan confidence > 0.5 akan ditampilkan

## Performance

Estimasi processing speed (RTX 3070/similar):
- 640x480 video: ~30-60 FPS
- 1080p video: ~15-30 FPS
- 4K video: ~5-10 FPS

Bergantung pada:
- Jumlah objects dalam frame
- Model complexity
- GPU specification

## License

Untuk keperluan internal.

## Contact

Untuk pertanyaan teknis atau bugs, silakan report ke development team.
