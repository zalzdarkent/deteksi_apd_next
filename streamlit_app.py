import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime
from PIL import Image
import tempfile
import os

# ====================
# PAGE CONFIG
# ====================
st.set_page_config(
    page_title="APD Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# UTILITY FUNCTIONS (from app.py)
# ====================

def load_config():
    """Load custom configuration from config.json"""
    try:
        if Path("config.json").exists():
            with open("config.json", 'r') as f:
                config = json.load(f)
                return config.get("custom", {})
    except Exception as e:
        st.warning(f"⚠️ Error loading config.json: {e}")
    return {"mask": True, "glove": True, "helm": True, "glasses": True, "boots": True}


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
    
    for cat in ["mask", "glove", "helm", "glasses", "boots"]:
        if cat in enabled_categories:
            is_compliant = results_dict.get(f"is_{cat}")
            if is_compliant is not None:  # Not null
                status = "OK" if is_compliant else "❌"
                status_full = f"{cat.upper()}: {status}"
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


def process_detections(frame, results, yolo_model, categories, excluded_labels=None):
    """
    Process YOLO detections and return:
    - Annotated frame with bounding boxes
    - Detection results dictionary
    - List of detection details for table
    """
    if excluded_labels is None:
        excluded_labels = {"safety cone", "safety vest", "no safety vest", "machinery", "vehicle", "no safety cone"}
    
    # Get detections
    boxes = results.boxes
    
    # Track best detection per category
    best_results = {}
    best_conf = {}
    detection_details = []
    
    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
        
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
    
    # Create results dictionary
    results_dict = {}
    for cat in ["mask", "glove", "helm", "glasses", "boots"]:
        if cat in categories:
            if cat in best_results:
                results_dict[f"is_{cat}"] = not best_results[cat].lower().startswith("no")
            else:
                results_dict[f"is_{cat}"] = False
        else:
            results_dict[f"is_{cat}"] = False
    
    # Draw APD status panel
    enabled_categories = set(categories)
    frame = draw_apd_status(frame, results_dict, enabled_categories)
    
    return frame, results_dict, detection_details


def process_image(image, yolo_model, categories, confidence_threshold):
    """Process image and return results"""
    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run YOLO inference
    results = yolo_model(frame, conf=confidence_threshold)[0]
    
    # Process detections
    annotated_frame, results_dict, detection_details = process_detections(
        frame, results, yolo_model, categories
    )
    
    # Convert back to RGB for display
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame_rgb, results_dict, detection_details


def process_video(video_path, yolo_model, categories, confidence_threshold):
    """Process video and return results"""
    cap = cv2.VideoCapture(video_path)
    
    frames_processed = []
    all_results = []
    frame_count = 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO inference
        results = yolo_model(frame, conf=confidence_threshold)[0]
        
        # Process detections
        annotated_frame, results_dict, detection_details = process_detections(
            frame, results, yolo_model, categories
        )
        
        # Convert to RGB
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frames_processed.append(annotated_frame_rgb)
        
        # Store results with frame number
        all_results.append({
            "frame": frame_count,
            "results": results_dict,
            "detections": detection_details
        })
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing... {frame_count}/{total_frames} frames")
    
    cap.release()
    
    progress_bar.empty()
    status_text.empty()
    
    return frames_processed, all_results


# ====================
# SIDEBAR CONFIGURATION
# ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Category selection
    st.subheader("Select APD Categories to Detect")
    default_config = load_config()
    
    categories = []
    if st.checkbox("👷 Mask", value=default_config.get("mask", True), key="cb_mask"):
        categories.append("mask")
    if st.checkbox("🧤 Glove", value=default_config.get("glove", True), key="cb_glove"):
        categories.append("glove")
    if st.checkbox("🪖 Helm", value=default_config.get("helm", True), key="cb_helm"):
        categories.append("helm")
    if st.checkbox("👓 Glasses", value=default_config.get("glasses", True), key="cb_glasses"):
        categories.append("glasses")
    if st.checkbox("👢 Boots", value=default_config.get("boots", True), key="cb_boots"):
        categories.append("boots")
    
    if not categories:
        st.warning("⚠️ Please select at least 1 category!")
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
    
    st.divider()
    st.info(f"✓ Selected: {', '.join([cat.upper() for cat in categories])}")


# ====================
# LOAD MODEL
# ====================
@st.cache_resource
def load_yolo_model():
    """Load YOLO model (cached)"""
    model_path = "model/yolo8_retrain_3x/best.pt"
    if not Path(model_path).exists():
        st.error(f"❌ Model not found at: {model_path}")
        st.stop()
    return YOLO(model_path)


yolo_model = load_yolo_model()


# ====================
# MAIN UI
# ====================
st.title("🔒 APD Detection System")
st.write("Upload images or videos to detect Personal Protective Equipment (APD) compliance")

# Create tabs
tab_image, tab_video = st.tabs(["📷 Image Detection", "🎬 Video Detection"])


# ====================
# IMAGE TAB
# ====================
with tab_image:
    st.subheader("Upload Image")
    
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
            st.image(image, use_column_width=True)
        
        with col2:
            st.write("**Detection Results**")
            
            # Process image
            with st.spinner("🔄 Processing image..."):
                annotated_image, results_dict, detection_details = process_image(
                    image, yolo_model, categories, confidence_threshold
                )
            
            st.image(annotated_image, use_column_width=True)
        
        # Display results
        st.divider()
        st.subheader("📊 Detection Summary")
        
        # Create results table
        summary_data = []
        for cat in ["mask", "glove", "helm", "glasses", "boots"]:
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
            use_container_width=True,
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
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("ℹ️ No detections found in the image")


# ====================
# VIDEO TAB
# ====================
with tab_video:
    st.subheader("Upload Video")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    if uploaded_video is not None:
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        try:
            # Process video
            with st.spinner("🔄 Processing video... This may take a while"):
                frames_processed, all_results = process_video(
                    video_path, yolo_model, categories, confidence_threshold
                )
            
            st.success(f"✅ Video processed! {len(frames_processed)} frames detected")
            
            # Display frame by frame or summary
            col_play, col_summary = st.columns([2, 1])
            
            with col_play:
                st.subheader("📹 Processed Video Preview")
                
                # Frame slider
                frame_idx = st.slider(
                    "Select frame to view",
                    min_value=0,
                    max_value=len(frames_processed) - 1,
                    step=1
                )
                
                # Display selected frame
                st.image(frames_processed[frame_idx], use_column_width=True)
                
                # Frame info
                current_result = all_results[frame_idx]
                st.write(f"**Frame {current_result['frame']}**")
            
            with col_summary:
                st.subheader("📊 Frame Results")
                
                # Current frame summary
                results_dict = current_result["results"]
                summary_data = []
                
                for cat in ["mask", "glove", "helm", "glasses", "boots"]:
                    if cat in categories:
                        status = results_dict.get(f"is_{cat}", False)
                        status_text = "✅" if status else "❌"
                        summary_data.append({
                            "Category": cat.upper(),
                            "Status": status_text
                        })
                
                st.dataframe(
                    summary_data,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Video statistics
            st.divider()
            st.subheader("📈 Video Statistics")
            
            # Analyze all frames
            total_frames = len(all_results)
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
                compliance_rate = (compliant_frames / total_frames * 100) if total_frames > 0 else 0
                
                stats_data.append({
                    "Category": cat.upper(),
                    "Compliant Frames": f"{compliant_frames}/{total_frames}",
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
                os.remove(video_path)


# ====================
# FOOTER
# ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>APD Detection System | Detection Model: YOLOv8</p>
    <p>Last Updated: February 2026</p>
</div>
""", unsafe_allow_html=True)
