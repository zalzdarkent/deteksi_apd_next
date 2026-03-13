# APD Detector Configuration

## Model Configuration
```yaml
model_path: "../model/yolov26_retrain/best.onnx"
input_size: 640x640
model_type: "yolov8"
```

## Detection Classes
```yaml
classes:
  - id: 0
    name: "helmet"
    color: [0, 255, 0]      # BGR format - Green
  - id: 1
    name: "person"
    color: [255, 0, 0]      # BGR format - Blue
  - id: 2
    name: "fire-extinguisher"
    color: [0, 0, 255]      # BGR format - Red
```

## Detection Thresholds
```yaml
confidence_threshold: 0.5          # Min confidence for detection
nms_iou_threshold: 0.45            # IoU threshold for NMS
```

## Input/Output
```yaml
input:
  video_directory: "../samples"
  video_formats: ["mp4", "avi", "mov", "mkv", "flv", "wmv"]

output:
  directory: "../output_videos"
  format: "mp4"
  codec: "h264"
  quality: "high"
```

## Performance Options
```yaml
preprocessing:
  resize_method: "LINEAR"           # OpenCV interpolation
  normalize: true                   # Normalize to 0-1
  channel_order: "RGB"              # Channel format

inference:
  batch_size: 1
  num_threads: 4                    # Intra-op threads
  optimization_level: "All"         # ONNX optimization
```

## Advanced Tuning

### Untuk video dengan banyak objects:
```yaml
confidence_threshold: 0.6
nms_iou_threshold: 0.5
```

### Untuk video dengan objects jarang:
```yaml
confidence_threshold: 0.4
nms_iou_threshold: 0.4
```

### Untuk performa optimal:
```yaml
preprocessing:
  resize_method: "AREA"             # Better untuk downsampling
  
inference:
  num_threads: 8                    # Sesuaikan dengan CPU cores
```
