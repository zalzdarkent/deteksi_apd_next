import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime
from PIL import Image, ImageOps
import tempfile
import os
import gspread
from google.oauth2.service_account import Credentials
import time
import random
# import base64 (Unused after Roboflow removal)

# ====================
# CONSTANTS & DIRECTORIES
# ====================
DATA_DIR = Path("data")
RECORDS_DIR = DATA_DIR / "records"
CONFIG_FILE = "config.json"
VIOLATIONS_DIR = Path("violations")
PASSED_DIR = Path("passed")
GOOGLE_SHEET_NAME = "deteksi-apd"
CREDENTIALS_FILE = "credentials.json"

DATA_DIR.mkdir(exist_ok=True)
RECORDS_DIR.mkdir(exist_ok=True)
VIOLATIONS_DIR.mkdir(exist_ok=True)
PASSED_DIR.mkdir(exist_ok=True)

# ====================
# PAGE CONFIG
# ====================
st.set_page_config(
    page_title="APD Detection System",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# UTILITY FUNCTIONS (from app.py)
# ====================

def load_config():
    try:
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get("areas", {})
    except Exception as e:
        st.warning(f"Error loading configuration: {e}")
    return {}

def get_google_sheet():
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        if not Path(CREDENTIALS_FILE).exists():
            return None
            
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open(GOOGLE_SHEET_NAME)
        return sh.sheet1
    except Exception as e:
        return None

def save_check_record(name, results_dict, area, last_state, last_save_time, debounce_seconds=2):
    state_key = f"{name}_{area}"
    time_key = f"{state_key}_time"
    current_time = time.time()
    
    current_state = json.dumps(results_dict, sort_keys=True)
    
    if current_state == last_state.get(state_key):
        return False
    
    last_save = last_save_time.get(time_key, 0)
    if current_time - last_save < debounce_seconds:
        return False
    
    worksheet = get_google_sheet()
    if worksheet is None:
        return False
    
    try:
        row_data = [area, name]
        
        for cat in ["helm", "fire-extinguisher"]:
            row_data.append(results_dict.get(f"is_{cat}", False))
        
        timestamp = datetime.now().isoformat()
        row_data.append(timestamp)
        
        worksheet.insert_row(row_data, index=2)
        
        last_state[state_key] = current_state
        last_save_time[time_key] = current_time
        
        return True
    except Exception as e:
        return False

def get_today_violation_dir():
    today = datetime.now().strftime("%Y-%m-%d")
    violation_dir = VIOLATIONS_DIR / today
    violation_dir.mkdir(parents=True, exist_ok=True)
    return violation_dir

def load_violation_data():
    data_file = VIOLATIONS_DIR / "data.json"
    
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            pass
    
    return {
        "violation_count": 0,
        "violations": []
    }

def save_violation_data(violation_data):
    data_file = VIOLATIONS_DIR / "data.json"
    
    try:
        with open(data_file, 'w') as f:
            json.dump(violation_data, f, indent=2)
        return True
    except Exception as e:
        return False


def save_violation_record(frame, name, results_dict, enabled_categories, last_violation_state):
    has_violation = any(not results_dict.get(f"is_{cat}") if cat in enabled_categories else False 
                       for cat in ["mask", "glove", "helm", "glasses", "boots"])
    
    if not has_violation:
        return False
    
    violation_key = f"{name}_{json.dumps(results_dict, sort_keys=True)}"
    
    if violation_key in last_violation_state:
        return False
    
    violation_data = load_violation_data()
    
    timestamp_ms = int(time.time() * 1000)
    random_id = random.randint(1000, 9999)
    unique_id = f"{timestamp_ms}_{random_id}"
    
    violation_detail = {}
    for cat in ["helm", "fire-extinguisher"]:
        if cat in enabled_categories:
            violation_detail[f"is_{cat}"] = results_dict.get(f"is_{cat}")
        else:
            violation_detail[f"is_{cat}"] = None
    
    violation_record = {
        "id": unique_id,
        "name": name,
        "area": results_dict.get("area", "Unknown"),
        "timestamp": datetime.now().isoformat(),
        "violations_detail": violation_detail,
        "path_image": f"violations/{datetime.now().strftime('%Y-%m-%d')}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    }
    
    violation_dir = get_today_violation_dir()
    img_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = violation_dir / img_filename
    
    try:
        cv2.imwrite(str(img_path), frame)
        
        violation_data["violations"].append(violation_record)
        violation_data["violation_count"] += 1
        
        save_violation_data(violation_data)
        
        last_violation_state[violation_key] = True
        
        return True
    except Exception as e:
        return False

def get_today_passed_dir():
    """Get or create passed directory for today"""
    today = datetime.now().strftime("%Y-%m-%d")
    passed_dir = PASSED_DIR / today
    passed_dir.mkdir(parents=True, exist_ok=True)
    return passed_dir

def load_passed_data():
    data_file = PASSED_DIR / "data.json"
    
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            pass
    
    return {
        "total_passed": 0,
        "passed": []
    }

def save_passed_data(passed_data):
    data_file = PASSED_DIR / "data.json"
    
    try:
        with open(data_file, 'w') as f:
            json.dump(passed_data, f, indent=2)
        return True
    except Exception as e:
        return False

def save_passed_record(frame, name, results_dict, enabled_categories, last_passed_state):
    has_violation = any(not results_dict.get(f"is_{cat}") if cat in enabled_categories else False 
                       for cat in ["mask", "glove", "helm", "glasses", "boots"])

    if has_violation:
        return False
    
    passed_key = f"{name}_{json.dumps(results_dict, sort_keys=True)}"
    
    if passed_key in last_passed_state:
        return False
    
    passed_data = load_passed_data()
    
    # Generate unique ID
    timestamp_ms = int(time.time() * 1000)
    random_id = random.randint(1000, 9999)
    unique_id = f"{timestamp_ms}_{random_id}"
    
    # Create passed record with null for disabled categories
    record_detail = {}
    for cat in ["helm", "fire-extinguisher"]:
        if cat in enabled_categories:
            record_detail[f"is_{cat}"] = results_dict.get(f"is_{cat}")
        else:
            record_detail[f"is_{cat}"] = None
    
    # Create passed record
    passed_record = {
        "id": unique_id,
        "name": name,
        "area": results_dict.get("area", "Unknown"),
        "timestamp": datetime.now().isoformat(),
        "compliant_detail": record_detail,
        "path_image": f"passed/{datetime.now().strftime('%Y-%m-%d')}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    }
    
    # Save screenshot
    passed_dir = get_today_passed_dir()
    img_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = passed_dir / img_filename
    
    try:
        cv2.imwrite(str(img_path), frame)
        
        # Add to passed data
        passed_data["passed"].append(passed_record)
        passed_data["total_passed"] += 1
        
        # Save passed data
        save_passed_data(passed_data)
        
        # Mark this passed state as recorded
        last_passed_state[passed_key] = True
        
        return True
    except Exception as e:
        return False

def normalize_label(label):
    """Normalize label untuk perbandingan"""
    return label.lower().replace("-", " ").strip()

def replace_label_names(label):
    """Ganti nama label Hardhat -> Helm"""
    if "Hardhat" in label:
        label = label.replace("Hardhat", "Helm")
    if "No-Hardhat" in label:
        label = label.replace("No-Hardhat", "No-Helm")
    return label

def get_label_color(label):
    """Tentukan warna label berdasarkan tipe (no = merah, lainnya = hijau)"""
    normalized = label.lower().replace("-", " ").replace("_", " ").strip()
    if normalized.startswith("no "):
        return (0, 0, 255)  # Red - tidak compliant
    else:
        return (0, 255, 0)   # Green - compliant

def draw_apd_status(frame, results_dict, enabled_categories):
    """Draw APD status labels on frame (list vertikal di corner)"""
    status_text = []
    
    for cat in ["helm", "fire-extinguisher"]:
        if cat in enabled_categories:
            is_compliant = results_dict.get(f"is_{cat}")
            if is_compliant is not None:  # Not null
                status = "COMPLIANT" if is_compliant else "VIOLATION"
                display_name = "APAR" if cat == "fire-extinguisher" else cat.upper()
                status_full = f"{display_name}: {status}"
                status_text.append((status_full, is_compliant))
    
    # Check if all passed
    all_passed = all(compliant for _, compliant in status_text)
    
    # Draw background box
    y_start = 20
    x_start = 10
    line_height = 25
    
    # Calculate box size
    max_text_width = 0
    for text, _ in status_text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        max_text_width = max(max_text_width, text_size[0])
    
    box_width = max_text_width + 20
    box_height = len(status_text) * line_height + 20
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start - 5, y_start - 5), 
                  (x_start + box_width, y_start + box_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw status labels
    for i, (text, is_compliant) in enumerate(status_text):
        y_pos = y_start + i * line_height + 15
        color = (0, 255, 0) if is_compliant else (0, 0, 255)
        cv2.putText(frame, text, (x_start + 5, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw PASSED label if all compliant
    if all_passed and status_text:
        passed_text = "PASSED"
        text_size = cv2.getTextSize(passed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        x_pos = frame.shape[1] - text_size[0] - 20
        y_pos = 40
        cv2.putText(frame, passed_text, (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return frame

def process_detections(frame, detections, categories, area="Unknown",
                       last_state=None, last_save_time=None, 
                       last_violation_state=None, last_passed_state=None,
                       excluded_labels=None):
    """
    Process detections list (standardized format) and return:
    - Annotated frame with bounding boxes
    - Detection results dictionary
    - List of detection details for table
    """
    if excluded_labels is None:
        excluded_labels = {"safety cone", "safety vest", "no safety vest", "machinery", "vehicle", "no safety cone"}
    
    # Init states if None
    if last_state is None: last_state = {}
    if last_save_time is None: last_save_time = {}
    if last_violation_state is None: last_violation_state = {}
    if last_passed_state is None: last_passed_state = {}

    # Initialize tracking variables
    best_results = {}
    best_conf = {}
    detection_details = []

    # Process detections
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        label = det['label']
        
        label = replace_label_names(label)
        norm = normalize_label(label)
        
        if norm in excluded_labels:
            continue
        
        # Check if this detection belongs to selected categories
        matched_category = None
        for category in categories:
            if category in label.lower():
                matched_category = category
                break
        
        if not matched_category:
            continue
            
        # Refinement: Category-specific confidence boosts/thresholds
        # Increase threshold for glasses to avoid background 'hallucinations'
        internal_threshold = 0.0
        if matched_category == "glasses":
            internal_threshold = 0.45  # Higher threshold for glasses specifically
            
        if conf < internal_threshold:
            continue
        
        # Keep only highest confidence result per category
        if matched_category not in best_conf or conf > best_conf[matched_category]:
            best_results[matched_category] = label
            best_conf[matched_category] = conf
        
        # Add to detection details
        detection_details.append({
            "category": matched_category,
            "label": label,
            "confidence": conf,
            "is_compliant": not label.lower().startswith("no")
        })
        
        # Draw bounding box
        color = get_label_color(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label inside bounding box
        display_text = f"{label} {conf:.2f}"
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + 5
        text_y = y1 + 20
        
        # Draw background for text
        cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 5), 
                    (text_x + text_size[0] + 2, text_y + 5), color, -1)
        cv2.putText(frame, display_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Create results dictionary for logging
    results_dict = {"area": area}
    for cat in ["helm", "fire-extinguisher"]:
        if cat in categories:
            if cat in best_results:
                results_dict[f"is_{cat}"] = not best_results[cat].lower().startswith("no")
            else:
                results_dict[f"is_{cat}"] = False
        else:
            results_dict[f"is_{cat}"] = False
    
    # 2. Logging and Records
    person_name = "Employee" # No face recognition, generic name or we could use area
    enabled_set = set(categories)
    
    # Save to Google Sheet
    save_check_record(person_name, results_dict, area, last_state, last_save_time)
    
    # Save violation screenshot
    save_violation_record(frame, person_name, results_dict, enabled_set, last_violation_state)
    
    # Save passed screenshot
    save_passed_record(frame, person_name, results_dict, enabled_set, last_passed_state)
    
    # Draw APD status panel on frame
    frame = draw_apd_status(frame, results_dict, enabled_set)
    
    return frame, results_dict, detection_details, person_name

def process_image(image, yolo_model, categories, area, confidence_threshold):
    """Process image and return results using local YOLO model"""
    # Fix orientation if image has EXIF rotation
    image = ImageOps.exif_transpose(image)
    
    # Ensure image is in RGB (handles PNG with alpha channel)
    image = image.convert("RGB")
    
    # Standardize dimension to improve detection on very high-res photos
    max_dim = 1024
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Convert to BGR for OpenCV drawing
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run Inference using local YOLO model
    results = yolo_model(frame, conf=confidence_threshold)[0]
    
    # Convert YOLO results to standard format
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
        
        detections.append({
            'box': [x1, y1, x2, y2],
            'confidence': conf,
            'label': label
        })
    
    # Process detections using the BGR frame for drawing
    annotated_frame, results_dict, detection_details, person_name = process_detections(
        frame, detections, categories, area=area,
        last_state=st.session_state.last_state,
        last_save_time=st.session_state.last_save_time,
        last_violation_state=st.session_state.last_violation_state,
        last_passed_state=st.session_state.last_passed_state
    )
    
    # Convert back to RGB for display
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame_rgb, results_dict, detection_details, person_name

def process_video(video_path, yolo_model, categories, area, confidence_threshold, placeholder=None, frame_skip=1):
    """Process video and return results with real-time streaming"""
    cap = cv2.VideoCapture(video_path)
    
    try:
        all_results = []
        frame_count = 0
        
        # Persistent state for video tracking
        last_state = {}
        last_save_time = {}
        last_violation_state = {}
        last_passed_state = {}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Standardize resolution for stability if video is very high-res
            max_dim = 1024
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
            # Skip processing if not the right frame
            if frame_count % frame_skip != 0 and frame_count != 1:
                if placeholder:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    placeholder.image(rgb_frame, width='stretch')
                continue

            # Run Inference using local YOLO model
            results = yolo_model(frame, conf=confidence_threshold)[0]
            
            # Convert YOLO results to standard format
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'label': label
                })
            
            # Process detections
            annotated_frame, results_dict, detection_details, person_name = process_detections(
                frame, detections, categories, area=area,
                last_state=last_state, last_save_time=last_save_time, 
                last_violation_state=last_violation_state, last_passed_state=last_passed_state
            )
            
            # Convert to RGB for streaming
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update streaming placeholder
            if placeholder:
                placeholder.image(annotated_frame_rgb, width='stretch')
            
            # Store results with frame number
            all_results.append({
                "frame": frame_count,
                "results": results_dict,
                "detections": detection_details,
                "person_name": person_name
            })
            
            # Update progress
            if frame_count % 10 == 0 or frame_count == total_frames:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing & Streaming... {frame_count}/{total_frames} frames")
        
        progress_bar.empty()
        status_text.empty()
        return all_results
    finally:
        cap.release()

# ====================
# SIDEBAR CONFIGURATION
# ====================
with st.sidebar:
    st.header("Control Panel")
    
    # Professional Navigation Menu
    st.divider()
    st.markdown("### Navigation")
    
    # Map display names to internal modes
    menu_options = {
        "Live Monitoring": "Live Detection",
        "Image Analysis": "Image Detection", 
        "Video Analytics": "Video Detection",
        "Security Records": "Security Records"
    }
    
    # Use material icons for a modern website look
    app_mode_display = st.radio(
        "Select Mode",
        options=list(menu_options.keys()),
        label_visibility="collapsed",
        captions=[":material/videocam:", ":material/image:", ":material/movie:", ":material/history:"]
    )
    app_mode = menu_options[app_mode_display]

    # Placement & Source Settings
    st.divider()
    st.markdown("### Placement Settings")
    areas_config = load_config()
    area_names = list(areas_config.keys()) if areas_config else ["Default"]
    
    selected_area = st.selectbox(
        "Detection Location",
        options=area_names,
        index=0
    )
    
    # CCTV Source (Only for Live Monitoring)
    camera_source = 0
    if app_mode == "Live Detection":
        st.markdown("#### Source Configuration")
        source_input = st.text_input("Camera Index or RTSP URL", value="0", help="Use '0' for default webcam or enter RTSP link for CCTV")
        try:
            camera_source = int(source_input)
        except ValueError:
            camera_source = source_input
            
        is_running = st.toggle("Activate Live Detection", value=False)
    
    # Category configuration
    st.divider()
    st.markdown("### Detection Categories")
    # Get configuration for selected area
    current_area_config = areas_config.get(selected_area, {
        "helm": True, "fire-extinguisher": True
    })
    
    categories = []
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Helm", value=current_area_config.get("helm", True), key=f"cb_helm_{selected_area}"):
            categories.append("helm")
    with col2:
        if st.checkbox("APAR (Fire Extinguisher)", value=current_area_config.get("fire-extinguisher", True), key=f"cb_fire_{selected_area}"):
            categories.append("fire-extinguisher")
    
    if not categories:
        st.warning("Please select at least one category for detection.")
        st.stop()
    
    # Confidence threshold
    st.subheader("Detection Threshold")
    confidence_threshold = st.slider(
        "Confidence Score",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Higher = more strict, lower = more detections"
    )
    
    # Frame Skip Configuration
    st.subheader("Performance Settings")
    frame_skip = st.slider(
        "Detection Frequency",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Process every N-th frame. Higher = faster but might miss brief moments."
    )
    st.caption(f"Processing 1/{frame_skip} frames")
    
    st.divider()
    st.info(f"Location: {selected_area}")
    st.info(f"Monitoring: {', '.join([cat.upper() for cat in categories])}")

@st.cache_resource
def load_local_model():
    """Initialize local YOLO11 model (cached)"""
    model_path = Path("model/yolov11_new_retrain/best.pt")
    if not model_path.exists():
        # Fallback to yolov11 2x if 3x is not found, to keep app running
        fallback_path = Path("model/yolov11_new_retrain/best.pt")
        if fallback_path.exists():
            st.warning(f"Model 3x not found at {model_path}. Using fallback 2x model.")
            return YOLO(str(fallback_path))
        else:
            st.error(f"Critical Error: YOLO model not found at {model_path}")
            return None
    return YOLO(str(model_path))

yolo_model = load_local_model()

# ====================
# MAIN UI
# ====================
st.markdown("<h1 style='text-align: center;'>APD Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Advanced Safety Monitoring with Real-time Detection and Logging</p>", unsafe_allow_html=True)

# Initialize session state for tracking
if 'last_state' not in st.session_state: st.session_state.last_state = {}
if 'last_save_time' not in st.session_state: st.session_state.last_save_time = {}
if 'last_violation_state' not in st.session_state: st.session_state.last_violation_state = {}
if 'last_passed_state' not in st.session_state: st.session_state.last_passed_state = {}

# ====================
# LIVE DETECTION
# ====================
if app_mode == "Live Detection":
    st.subheader("Real-time CCTV Detection")
    
    if is_running:
        cap = cv2.VideoCapture(camera_source)
        try:
            # Create persistent placeholders
            alert_holder = st.empty()
            st_frame = st.empty()
            
            # UI for stopping within the main area
            stop_btn = st.button("Stop Stream", type="primary")
            
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to connect to camera source.")
                    break
                
                # Frame skipping for live monitoring
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_idx % frame_skip != 0 and frame_idx != 0:
                    # Just show the raw frame to keep the stream smooth
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(rgb_frame, width='stretch')
                    continue

                # Run inference using local YOLO model
                results = yolo_model(frame, conf=confidence_threshold)[0]
                
                # Convert YOLO results to standard format
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': conf,
                        'label': label
                    })
                
                # Process detections
                annotated_frame, results_dict, detection_details, person_name = process_detections(
                    frame, detections, categories, area=selected_area,
                    last_state=st.session_state.last_state,
                    last_save_time=st.session_state.last_save_time,
                    last_violation_state=st.session_state.last_violation_state,
                    last_passed_state=st.session_state.last_passed_state
                )
                
                # Persistent Alerts logic
                missing = [cat.upper() if cat != "fire-extinguisher" else "APAR" 
                          for cat in categories if not results_dict.get(f"is_{cat}", False)]
                if missing:
                    alert_holder.error(f"⚠️ **VIOLASI KEAMANAN: {', '.join(missing)} TIDAK TERDETEKSI!**")
                else:
                    alert_holder.success("✅ **SEMUA APD LENGKAP - RADIUS AMAN**")
                
                # Convert to RGB
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Stream to UI
                st_frame.image(annotated_rgb, width='stretch')
            
            st.info("Stream stopped.")
        finally:
            cap.release()
    else:
        st.info("Please toggle 'Start Detection' in the sidebar to begin CCTV monitoring.")

# ====================
# IMAGE DETECTION
# ====================
elif app_mode == "Image Detection":
    st.subheader("Image Analysis")
    
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_image is not None:
        # Load and display original image
        image = Image.open(uploaded_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Image**")
            st.image(image, width='stretch')
        
        with col2:
            st.write("**Detection Results**")
            
            # Process image
            with st.spinner("Processing image..."):
                annotated_image, results_dict, detection_details, person_name = process_image(
                    image, yolo_model, categories, selected_area, confidence_threshold
                )
            
            # Show Alerts for Image Detection
            missing = [cat.upper() if cat != "fire-extinguisher" else "APAR" 
                      for cat in categories if not results_dict.get(f"is_{cat}", False)]
            if missing:
                st.error(f"⚠️ **VIOLASI: {', '.join(missing)} TIDAK TERDETEKSI!**")
            else:
                st.success("✅ **SEMUA APD TERDETEKSI**")

            st.image(annotated_image, width='stretch')
        
        # Display results
        st.divider()
        st.subheader("📊 Detection Summary")
        
        # Create results table
        summary_data = []
        for cat in ["helm", "fire-extinguisher"]:
            if cat in categories:
                status = results_dict.get(f"is_{cat}", False)
                status_text = "✅ COMPLIANT" if status else "❌ VIOLATION"
                summary_data.append({
                    "Category": cat.upper(),
                    "Status": status_text,
                    "Compliance": "Yes" if status else "No"
                })
        
        st.dataframe(
            summary_data,
            width='stretch',
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn(width="medium"),
                "Status": st.column_config.TextColumn(width="medium"),
                "Compliance": st.column_config.TextColumn(width="medium")
            }
        )
        
        # Display detection details
        if detection_details:
            st.subheader("🔍 Detection Details")
            
            details_data = []
            for det in detection_details:
                details_data.append({
                    "Category": det["category"].upper(),
                    "Label": det["label"],
                    "Confidence": f"{det['confidence']:.2%}",
                    "Compliant": "✅" if det["is_compliant"] else "❌"
                })
            
            st.dataframe(
                details_data,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("ℹ️ No detections found in the image")

# ====================
# SECURITY RECORDS
# ====================
elif app_mode == "Security Records": # Security Records
    st.subheader("Security Audit Records")
    
    tab_violations, tab_passed = st.tabs(["Violation History", "Passed History"])
    
    with tab_violations:
        violation_data = load_violation_data()
        records = violation_data.get("violations", [])
        
        if not records:
            st.info("No violation records found.")
        else:
            # Sort by timestamp descending
            records = sorted(records, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            for rec in records:
                # Format timestamp for better readability
                try:
                    dt_obj = datetime.fromisoformat(rec['timestamp'])
                    formatted_time = dt_obj.strftime("%d %b %Y, %H:%M:%S")
                except:
                    formatted_time = rec['timestamp']
                    
                with st.expander(f"Violation: {rec['area']} | {formatted_time}"):
                    col_info, col_img = st.columns([1, 2])
                    with col_info:
                        st.markdown("#### Violation Details")
                        # Fix key from 'details' to 'violations_detail'
                        details = rec.get("violations_detail", {})
                        
                        violations_found = []
                        for cat, status in details.items():
                            category_name = cat.replace("is_", "").upper()
                            if status is False:
                                st.error(f"MISSING: {category_name}")
                                violations_found.append(category_name)
                            elif status is True:
                                st.success(f"OK: {category_name}")
                        
                        if not violations_found:
                            st.warning("No specific categories recorded as missing.")
                            
                        st.divider()
                        st.caption(f"ID: {rec['id']}")
                        st.caption(f"Name: {rec['name']}")
                    
                    with col_img:
                        img_path = Path("") / rec['path_image']
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.error(f"Image not found at: {img_path}")
                            
    with tab_passed:
        passed_data = load_passed_data()
        records = passed_data.get("passed", [])
        
        if not records:
            st.info("No passed records found.")
        else:
            # Sort by timestamp descending
            records = sorted(records, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            for rec in records:
                try:
                    dt_obj = datetime.fromisoformat(rec['timestamp'])
                    formatted_time = dt_obj.strftime("%d %b %Y, %H:%M:%S")
                except:
                    formatted_time = rec['timestamp']
                    
                with st.expander(f"Passed: {rec['area']} | {formatted_time}"):
                    col_info, col_img = st.columns([1, 2])
                    with col_info:
                        st.success("All APD Compliant")
                        st.divider()
                        st.caption(f"ID: {rec['id']}")
                        st.caption(f"Name: {rec['name']}")
                    
                    with col_img:
                        img_path = Path("") / rec['path_image']
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.error(f"Image not found at: {img_path}")

# ====================
# VIDEO DETECTION
# ====================
elif app_mode == "Video Detection":
    st.subheader("Video Analytics")
    
    # Sample videos in samples/ folder
    sample_videos = list(Path("samples").glob("*.mp4"))
    sample_options = ["None (Upload your own)"] + [v.name for v in sample_videos]
    
    selected_sample = st.selectbox(
        "Or choose a sample video",
        options=sample_options,
        index=0
    )
    
    uploaded_video = None
    if selected_sample == "None (Upload your own)":
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"]
        )
    else:
        # Load from samples
        uploaded_video = open(Path("samples") / selected_sample, "rb")

    if uploaded_video is not None:
        try:
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            try:
                # Placeholders for video alert and stream
                alert_holder = st.empty()
                video_placeholder = st.empty()
                
                # Process video with real-time streaming
                with st.spinner(" Processing & Streaming video..."):
                    processed_frames = 0
                    cap_video = cv2.VideoCapture(video_path)
                    total_v_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap_video.release()

                    # We'll update the alert_holder inside process_video if we refactor it, 
                    # but for now we'll handle the last state from all_results
                    all_results = process_video(
                        video_path, yolo_model, categories, selected_area, 
                        confidence_threshold, placeholder=video_placeholder,
                        frame_skip=frame_skip
                    )
                
                if all_results:
                    last_res = all_results[-1]["results"]
                    missing = [cat.upper() if cat != "fire-extinguisher" else "APAR" 
                              for cat in categories if not last_res.get(f"is_{cat}", False)]
                    if missing:
                        alert_holder.error(f"⚠️ **STATUS AKHIR: {', '.join(missing)} TIDAK TERDETEKSI!**")
                    else:
                        alert_holder.success("✅ **STATUS AKHIR: SEMUA APD LENGKAP**")

                st.success(f"✅ Video processed! {len(all_results)} frames analyzed")
                
                # Show summary of violations found
                with st.expander("Detailed Statistics", expanded=True):
                    # Analyze all frames
                    total_frames_analyzed = len(all_results)
                    compliance_by_category = {cat: [] for cat in categories}
                    
                    for frame_result in all_results:
                        for cat in categories:
                            compliance_by_category[cat].append(
                                frame_result["results"].get(f"is_{cat}", False)
                            )
                    
                    # Create statistics table
                    stats_data = []
                    for cat in categories:
                        compliant_frames = sum(compliance_by_category[cat])
                        compliance_rate = (compliant_frames / total_frames_analyzed * 100) if total_frames_analyzed > 0 else 0
                        
                        display_name = "APAR" if cat == "fire-extinguisher" else cat.upper()
                        stats_data.append({
                            "Category": display_name,
                            "Compliant Frames": f"{compliant_frames}/{total_frames_analyzed}",
                            "Compliance Rate": f"{compliance_rate:.1f}%",
                            "Status": "✅ GOOD" if compliance_rate >= 80 else "⚠️ NEEDS ATTENTION"
                        })
                    
                    st.dataframe(
                        stats_data,
                        use_container_width=True,
                        hide_index=True
                    )
            finally:
                # Cleanup temporary file
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except Exception as e:
                        # Log warning instead of crashing on cleanup failure
                        pass
        finally:
            if selected_sample != "None (Upload your own)":
                uploaded_video.close()

# ====================
# FOOTER
# ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>APD Detection System | Engine: Local YOLO11 Model</p>
    <p>Last Updated: March 2026</p>
</div>
""", unsafe_allow_html=True)