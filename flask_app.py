import os
import cv2
import threading
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join('model', 'yolov11_roboflow', 'weights.pt')
model = YOLO(MODEL_PATH)

# Video Samples Directory
SAMPLES_DIR = 'samples'

def gen_frames(video_name, start_frame=0):
    video_path = os.path.join(SAMPLES_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Get original FPS to maintain timing
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run YOLOv11 inference
        results = model.predict(frame, conf=0.5, verbose=False)
        detections = results[0].boxes
        
        persons = []
        helmets = []
        others = []
        
        # Organize detections
        for box in detections:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            
            if cls == 2: # Person
                persons.append({'box': coords, 'conf': conf, 'has_helm': False})
            elif cls == 1: # Helm
                helmets.append({'box': coords, 'conf': conf, 'used': False})
            else: # Others (fire-extinguisher)
                others.append({'box': coords, 'conf': conf, 'cls': cls})
                
        # Association Logic: Check if helm is inside or overlaps person
        for p in persons:
            px1, py1, px2, py2 = p['box']
            for h in helmets:
                hx1, hy1, hx2, hy2 = h['box']
                # Check if helmet center is inside person box
                h_center_x = (hx1 + hx2) / 2
                h_center_y = (hy1 + hy2) / 2
                
                if px1 <= h_center_x <= px2 and py1 <= h_center_y <= py2:
                    p['has_helm'] = True
                    h['used'] = True
                    # Optimization: once a person has a helm, move to next person
                    break
        
        # Draw Custom Boxes
        annotated_frame = frame.copy()
        
        # Draw Persons
        for p in persons:
            x1, y1, x2, y2 = map(int, p['box'])
            color = (0, 255, 0) if p['has_helm'] else (0, 255, 255) # Green vs Yellow (BGR)
            label = "Safe" if p['has_helm'] else "No Helm"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            # Add status text
            cv2.putText(annotated_frame, f"{label} {p['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
        # Draw Unused Helmets (Helmets not on people)
        for h in helmets:
            if not h['used']:
                x1, y1, x2, y2 = map(int, h['box'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "Helm Only", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
        # Draw Others (Fire Extinguisher)
        for o in others:
            x1, y1, x2, y2 = map(int, o['box'])
            name = model.names[o['cls']]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue
            cv2.putText(annotated_frame, f"{name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Encode frame to JPG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/')
def index():
    videos = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    selected_video = request.args.get('video')
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
        
    results = model.predict(frame, conf=0.5, verbose=False)
    detections = results[0].boxes
    
    persons = []
    helmets = []
    others = []
    
    for box in detections:
        cls = int(box.cls[0])
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        if cls == 2: persons.append({'box': coords, 'conf': conf, 'has_helm': False})
        elif cls == 1: helmets.append({'box': coords, 'conf': conf, 'used': False})
        else: others.append({'box': coords, 'conf': conf, 'cls': cls})
            
    for p in persons:
        px1, py1, px2, py2 = p['box']
        for h in helmets:
            hx1, hy1, hx2, hy2 = h['box']
            h_center_x = (hx1 + hx2) / 2
            h_center_y = (hy1 + hy2) / 2
            if px1 <= h_center_x <= px2 and py1 <= h_center_y <= py2:
                p['has_helm'] = True
                h['used'] = True
                break
    
    annotated_frame = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = map(int, p['box'])
        color = (0, 255, 0) if p['has_helm'] else (0, 255, 255)
        label = "Safe" if p['has_helm'] else "No Helm"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated_frame, f"{label} {p['conf']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
    for h in helmets:
        if not h['used']:
            x1, y1, x2, y2 = map(int, h['box'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Helm Only", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
    for o in others:
        x1, y1, x2, y2 = map(int, o['box'])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
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
    # Run Flask on all interfaces to allow local network testing if needed
    app.run(host='0.0.0.0', port=5000, threaded=True)
