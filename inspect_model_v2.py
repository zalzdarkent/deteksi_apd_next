from ultralytics import YOLO
import sys

try:
    model_path = "model/yolov11_new_retrain/best.pt"
    model = YOLO(model_path)
    print(f"Model: {model_path}")
    print(f"Classes: {model.names}")
except Exception as e:
    print(f"Error: {e}")
