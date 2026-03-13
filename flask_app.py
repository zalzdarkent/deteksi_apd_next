import os
import cv2
import time
import threading
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'yolov26', 'weights.onnx')
model = YOLO(MODEL_PATH, task='detect')

SAMPLES_DIR = 'samples'

def gen_frames(video_name, start_frame=0):
    video_path = os.path.join(SAMPLES_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 60
    frame_delay = 1.0 / fps
    
    prev_time = time.time()
    
    while cap.isOpened():
        current_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        elapsed = current_time - prev_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)
        elif elapsed > frame_delay * 2:
            skip_count = int(elapsed / frame_delay) - 1
            for _ in range(min(skip_count, 5)):
                cap.grab()
        
        prev_time = time.time()

        results = model.predict(frame, conf=0.5, imgsz=640, verbose=False, half=False)
        detections = results[0].boxes
        
        persons = []
        helmets = []
        others = []
        
        if len(detections) > 0:
            boxes = detections.xyxy.cpu().numpy()
            clss = detections.cls.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                cls = int(clss[i])
                coords = boxes[i].tolist()
                conf = float(confs[i])
                
                if cls == 2: 
                    persons.append({'box': coords, 'conf': conf, 'has_helm': False})
                elif cls == 1:
                    helmets.append({'box': coords, 'conf': conf, 'used': False})
                else: 
                    name = model.names[cls] if cls in model.names else f"ID {cls}"
                    others.append({'box': coords, 'conf': conf, 'name': name})
                    
        # Association Logic
        for p in persons:
            px1, py1, px2, py2 = p['box']
            for h in helmets:
                if h['used']: continue
                hx1, hy1, hx2, hy2 = h['box']
                h_center_x = (hx1 + hx2) / 2
                h_center_y = (hy1 + hy2) / 2
                
                if px1 <= h_center_x <= px2 and py1 <= h_center_y <= py2:
                    p['has_helm'] = True
                    h['used'] = True
                    break
        
        # Draw Results
        annotated_frame = frame.copy()
        
        for p in persons:
            x1, y1, x2, y2 = map(int, p['box'])
            color = (0, 255, 0) if p['has_helm'] else (0, 255, 255)
            label = "Safe" if p['has_helm'] else "No Helm"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated_frame, f"{label} {p['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
        for h in helmets:
            if not h['used']:
                x1, y1, x2, y2 = map(int, h['box'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(annotated_frame, "Helm Only", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            
        for o in others:
            x1, y1, x2, y2 = map(int, o['box'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, o['name'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode & Yield
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
               
    cap.release()

@app.route('/')
def index():
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)
    videos = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    selected_video = request.args.get('video')
    if not selected_video and videos:
        selected_video = videos[0]
    return render_template('index.html', videos=videos, selected_video=selected_video)

@app.route('/video_info/<video_name>')
def video_info(video_name):
    video_path = os.path.join(SAMPLES_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 404
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    duration = total_frames / fps
    cap.release()
    
    return jsonify({
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration
    })

def get_annotated_frame(video_name, frame_idx):
    video_path = os.path.join(SAMPLES_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        return None
        
    results = model.predict(frame, conf=0.5, imgsz=640, verbose=False, half=False)
    detections = results[0].boxes
    
    # Reuse drawing logic or keep it simple for scrub/pause
    annotated_frame = frame.copy()
    if len(detections) > 0:
        boxes = detections.xyxy.cpu().numpy()
        clss = detections.cls.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cls_name = model.names[int(clss[i])]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    return buffer.tobytes()

@app.route('/video_frame/<video_name>')
def video_frame(video_name):
    if video_name not in os.listdir(SAMPLES_DIR):
        return "Video not found", 404
    
    frame_idx = request.args.get('frame', 0, type=int)
    frame_bytes = get_annotated_frame(video_name, frame_idx)
    
    if frame_bytes is None:
        return "Frame not found", 404
        
    return Response(frame_bytes, mimetype='image/jpeg')

@app.route('/video_feed/<video_name>')
def video_feed(video_name):
    if video_name not in os.listdir(SAMPLES_DIR):
        return "Video not found", 404
    
    start_frame = request.args.get('start_frame', 0, type=int)
    return Response(gen_frames(video_name, start_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
