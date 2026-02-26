from ultralytics import YOLO
from pathlib import Path

model_path = "model/yolo8_retrain_3x/best.pt"
if not Path(model_path).exists():
    print(f"Model not found at: {model_path}")
    exit(1)

model = YOLO(model_path)
print("--- Model Information ---")
print(f"Model Path: {model_path}")
print(f"Model Names: {model.names}")
print(f"Task: {model.task}")
if hasattr(model, 'overrides'):
    print(f"Overrides: {model.overrides}")
