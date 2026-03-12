from ultralytics import YOLO

# Load model PyTorch kamu
model = YOLO("model/yolov26_retrain/best.pt")

# Export ke format ONNX
# imgsz: sesuaikan dengan ukuran gambar saat training (misal 640)
# dynamic: aktifkan jika resolusi kamera CCTV bisa berubah-ubah
path_onnx = model.export(format="onnx", imgsz=640, dynamic=True)

print(f"Model berhasil diubah ke: {path_onnx}")