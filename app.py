import cv2
import json
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# Create necessary directories
DATA_DIR = Path("data")
FACES_DIR = DATA_DIR / "faces"
RECORDS_DIR = DATA_DIR / "records"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
CONFIG_FILE = "config.json"

# Violation tracking directories
VIOLATIONS_DIR = Path("violations")
PASSED_DIR = Path("passed")
GOOGLE_SHEET_NAME = "deteksi-apd"
CREDENTIALS_FILE = "credentials.json" 

DATA_DIR.mkdir(exist_ok=True)   
FACES_DIR.mkdir(exist_ok=True)
RECORDS_DIR.mkdir(exist_ok=True)
VIOLATIONS_DIR.mkdir(exist_ok=True)
PASSED_DIR.mkdir(exist_ok=True)

# YOLO Models
yolo_model = YOLO('model/yolo8_retrain/best.pt')

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_config():
    """Load custom configuration from config.json"""
    try:
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get("custom", {})
    except Exception as e:
        print(f"Warning: Error loading config.json: {e}")
    return {"mask": True, "glove": True, "helm": True, "glasses": True, "boots": True}

def load_embeddings():
    """Load face embeddings from JSON"""
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_embeddings(embeddings):
    """Save face embeddings to JSON"""
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(embeddings, f, indent=2)

def get_face_embedding(frame, face_coords):
    """Extract face embedding using deepface"""
    try:
        x, y, w, h = face_coords
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None
            
        # Get embedding
        embedding = DeepFace.represent(face_roi, model_name="Facenet512", enforce_detection=False)
        if embedding:
            return embedding[0]["embedding"]
    except Exception as e:
        print(f"Error extracting embedding: {e}")
    return None

def cosine_distance(a, b):
    """Calculate cosine distance between two embeddings"""
    a = np.array(a)
    b = np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def recognize_face(frame, face_coords, embeddings, threshold=0.35):
    """Recognize face by comparing embeddings - MORE STRICT"""
    embedding = get_face_embedding(frame, face_coords)
    if embedding is None:
        return None
    
    best_match = None
    best_distance = threshold
    
    for name, stored_embedding in embeddings.items():
        distance = cosine_distance(embedding, stored_embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = name
    
    return best_match

def register_face(name):
    """Register a new person by capturing face"""
    cap = cv2.VideoCapture(0)
    collected = 0
    embeddings = load_embeddings()
    collected_embeddings = []
    
    print(f"\nCapturing face for: {name}")
    print("Press 'c' to capture, 'q' to finish")
    print("(Capture 5-10 samples dari berbagai angle untuk hasil lebih akurat)")
    
    while collected < 10:  # Collect 10 samples for better accuracy
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {collected}/10", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and len(faces) > 0:
            x, y, w, h = faces[0]
            embedding = get_face_embedding(frame, (x, y, w, h))
            if embedding:
                collected += 1
                collected_embeddings.append(embedding)
                print(f"Captured {collected}/10")
                # Save the face image
                img_path = FACES_DIR / f"{name}_{collected}.jpg"
                cv2.imwrite(str(img_path), frame[y:y+h, x:x+w])
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if collected >= 5:  # Require at least 5 samples
        # Calculate average embedding from all samples
        avg_embedding = np.mean(collected_embeddings, axis=0).tolist()
        embeddings[name] = avg_embedding
        save_embeddings(embeddings)
        print(f"✓ {name} registered successfully with {collected} samples!")
        input("Tekan Enter untuk kembali...")
        return True
    else:
        print(f"❌ Need at least 5 samples! (Collected: {collected})")
        input("Tekan Enter untuk kembali...")
        return False

def get_category_selection():
    """Menu untuk memilih kategori yang mau dicek"""
    print("\n=== Pilih Kategori APD ===")
    print("1. Masker (Mask/No-Mask)")
    print("2. Sarung Tangan (Glove/No-Glove)")
    print("3. Helm (Helm/No-Helm)")
    choice = input("Pilih yang mau dicek (1/2/3): ").strip()
    
    category_map = {
        "1": "mask",
        "2": "glove",
        "3": "helm"
    }
    
    selected_category = category_map.get(choice)
    if not selected_category:
        print("Pilihan tidak valid!")
        return None
    
    return selected_category

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
    # Normalize label: lowercase dan ganti - atau _ dengan spasi
    normalized = label.lower().replace("-", " ").replace("_", " ").strip()
    # Check kalau dimulai dengan "no "
    if normalized.startswith("no "):
        return (0, 0, 255)  # Red - tidak compliant
    else:
        return (0, 255, 0)   # Green - compliant

def get_google_sheet():
    """Connect ke Google Sheet menggunakan service account"""
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open(GOOGLE_SHEET_NAME)
        return sh.sheet1  # Return first worksheet
    except FileNotFoundError:
        print(f"❌ Error: {CREDENTIALS_FILE} tidak ditemukan!")
        print("Setup Google Sheets:")
        print("1. Buka https://console.cloud.google.com/")
        print("2. Buat service account dan download JSON credentials")
        print("3. Rename ke 'credentials.json' dan letakkan di folder project")
        print("4. Buat Google Sheet dengan nama: APD Records")
        print("5. Share sheet dengan email service account")
        return None
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"❌ Error: Google Sheet '{GOOGLE_SHEET_NAME}' tidak ditemukan!")
        print("Pastikan sudah membuat Google Sheet dan share dengan service account")
        return None

def save_check_record(name, results_dict, last_state, last_save_time, debounce_seconds=2):
    """Save APD check result to Google Sheet only if ANY state changed AND debounce time passed"""
    import time
    
    state_key = f"{name}"
    time_key = f"{state_key}_time"
    current_time = time.time()
    
    # Create state string for comparison
    current_state = json.dumps(results_dict, sort_keys=True)
    
    # Check if state is same as last time
    if current_state == last_state.get(state_key):
        return False
    
    # Check if enough time has passed since last save (debounce)
    last_save = last_save_time.get(time_key, 0)
    if current_time - last_save < debounce_seconds:
        return False
    
    # Get Google Sheet
    worksheet = get_google_sheet()
    if worksheet is None:
        return False
    
    try:
        # Prepare record
        row_data = [name]
        
        # Add all category values in order: mask, glove, helm, glasses, boots
        for cat in ["mask", "glove", "helm", "glasses", "boots"]:
            row_data.append(results_dict.get(f"is_{cat}", False))
        
        timestamp = datetime.now().isoformat()
        row_data.append(timestamp)
        
        # Insert row at top (row 2, right after header)
        worksheet.insert_row(row_data, index=2)
        
        # Update last state and time
        last_state[state_key] = current_state
        last_save_time[time_key] = current_time
        
        print(f"✓ Record saved: {name}")
        return True
    except Exception as e:
        print(f"❌ Error saving record: {e}")
        return False

def get_today_violation_dir():
    """Get or create violation directory for today"""
    today = datetime.now().strftime("%Y-%m-%d")
    violation_dir = VIOLATIONS_DIR / today
    violation_dir.mkdir(parents=True, exist_ok=True)
    return violation_dir

def load_violation_data():
    """Load violation data from JSON"""
    data_file = VIOLATIONS_DIR / "data.json"
    
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading violation data: {e}")
    
    # Create new violation data structure
    return {
        "violation_count": 0,
        "violations": []
    }

def save_violation_data(violation_data):
    """Save violation data to JSON"""
    data_file = VIOLATIONS_DIR / "data.json"
    
    try:
        with open(data_file, 'w') as f:
            json.dump(violation_data, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving violation data: {e}")
        return False

def save_violation_record(frame, name, results_dict, enabled_categories, last_violation_state):
    """Save violation record with screenshot if any APD category is False"""
    import time
    import random
    
    # Check if there's any violation (any False value in enabled categories)
    has_violation = any(not results_dict.get(f"is_{cat}") if cat in enabled_categories else False 
                       for cat in ["mask", "glove", "helm", "glasses", "boots"])
    
    if not has_violation:
        return False
    
    # Create unique key for this person's current violation state
    violation_key = f"{name}_{json.dumps(results_dict, sort_keys=True)}"
    
    # Check if this exact violation was already recorded (avoid duplicate screenshots)
    if violation_key in last_violation_state:
        return False
    
    # Load current violation data
    violation_data = load_violation_data()
    
    # Generate unique ID
    timestamp_ms = int(time.time() * 1000)
    random_id = random.randint(1000, 9999)
    unique_id = f"{timestamp_ms}_{random_id}"
    
    # Create violation record with null for disabled categories
    violation_detail = {}
    for cat in ["mask", "glove", "helm", "glasses", "boots"]:
        if cat in enabled_categories:
            violation_detail[f"is_{cat}"] = results_dict.get(f"is_{cat}")
        else:
            violation_detail[f"is_{cat}"] = None
    
    # Create violation record
    violation_record = {
        "id": unique_id,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "violations_detail": violation_detail,
        "path_image": f"violations/{datetime.now().strftime('%Y-%m-%d')}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    }
    
    # Save screenshot
    violation_dir = get_today_violation_dir()
    img_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = violation_dir / img_filename
    
    try:
        cv2.imwrite(str(img_path), frame)
        
        # Add to violation data
        violation_data["violations"].append(violation_record)
        violation_data["violation_count"] += 1
        
        # Save violation data
        save_violation_data(violation_data)
        
        # Mark this violation state as recorded
        last_violation_state[violation_key] = True
        
        print(f"⚠️  Violation saved: {name}")
        return True
    except Exception as e:
        print(f"❌ Error saving violation: {e}")
        return False

def get_today_passed_dir():
    """Get or create passed directory for today"""
    today = datetime.now().strftime("%Y-%m-%d")
    passed_dir = PASSED_DIR / today
    passed_dir.mkdir(parents=True, exist_ok=True)
    return passed_dir

def load_passed_data():
    """Load passed data from JSON"""
    data_file = PASSED_DIR / "data.json"
    
    if data_file.exists():
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading passed data: {e}")
    
    # Create new passed data structure
    return {
        "passed_count": 0,
        "records": []
    }

def save_passed_data(passed_data):
    """Save passed data to JSON"""
    data_file = PASSED_DIR / "data.json"
    
    try:
        with open(data_file, 'w') as f:
            json.dump(passed_data, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving passed data: {e}")
        return False

def draw_apd_status(frame, results_dict, enabled_categories):
    """Draw APD status labels on frame (list vertikal di corner)"""
    # Define category symbols and colors
    status_text = []
    
    for cat in ["mask", "glove", "helm", "glasses", "boots"]:
        if cat in enabled_categories:
            is_compliant = results_dict.get(f"is_{cat}")
            if is_compliant is not None:  # Not null
                status = "OK" if is_compliant else "NO"
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

def save_passed_record(frame, name, results_dict, enabled_categories, last_passed_state):
    """Save passed record (no violations) with screenshot if all APD categories pass"""
    import time
    import random
    
    # Check if there's any violation (any False value in enabled categories)
    has_violation = any(not results_dict.get(f"is_{cat}") if cat in enabled_categories else False 
                       for cat in ["mask", "glove", "helm", "glasses", "boots"])
    
    if has_violation:
        return False
    
    # Create unique key for this person's current passed state
    passed_key = f"{name}_{json.dumps(results_dict, sort_keys=True)}"
    
    # Check if this exact passed state was already recorded (avoid duplicate screenshots)
    if passed_key in last_passed_state:
        return False
    
    # Load current passed data
    passed_data = load_passed_data()
    
    # Generate unique ID
    timestamp_ms = int(time.time() * 1000)
    random_id = random.randint(1000, 9999)
    unique_id = f"{timestamp_ms}_{random_id}"
    
    # Create passed record with null for disabled categories
    record_detail = {}
    for cat in ["mask", "glove", "helm", "glasses", "boots"]:
        if cat in enabled_categories:
            record_detail[f"is_{cat}"] = results_dict.get(f"is_{cat}")
        else:
            record_detail[f"is_{cat}"] = None
    
    # Create passed record
    passed_record = {
        "id": unique_id,
        "name": name,
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
        passed_data["records"].append(passed_record)
        passed_data["passed_count"] += 1
        
        # Save passed data
        save_passed_data(passed_data)
        
        # Mark this passed state as recorded
        last_passed_state[passed_key] = True
        
        print(f"✓ Passed record saved: {name}")
        return True
    except Exception as e:
        print(f"❌ Error saving passed record: {e}")
        return False

def check_apd():
    """Main check APD - pilih area"""
    while True:
        clear_screen()
        print("\n" + "="*40)
        print("PILIH AREA")
        print("="*40)
        print("1. Casting (Check: Mask, Glove, Helm, Glasses, Boots)")
        print("2. Machining (Check: Glove, Helm, Glasses, Boots)")
        print("3. Custom (Check: Based on config.json)")
        print("4. Kembali ke Menu Utama")
        choice = input("Pilih (1/2/3/4): ").strip()
        
        if choice == "1":
            clear_screen()
            check_apd_area("casting", ["mask", "glove", "helm", "glasses", "boots"])
            break
        elif choice == "2":
            clear_screen()
            check_apd_area("machining", ["glove", "helm", "glasses", "boots"])
            break
        elif choice == "3":
            clear_screen()
            config = load_config()
            custom_categories = [cat for cat, enabled in config.items() if enabled]
            if not custom_categories:
                print("❌ Tidak ada kategori yang di-enable di config.json!")
                print("Pastikan minimal 1 kategori di-set ke true dalam config.json")
                input("Tekan Enter untuk kembali...")
                continue
            categories_display = ", ".join([c.upper() for c in custom_categories])
            print(f"Custom Categories Enabled: {categories_display}")
            input("Tekan Enter untuk memulai...")
            clear_screen()
            check_apd_area("custom", custom_categories)
            break
        elif choice == "4":
            clear_screen()
            return
        else:
            print("❌ Pilihan tidak valid!")
            input("Tekan Enter untuk lanjut...")

def check_apd_area(area, categories):
    """Check APD untuk spesifik area dengan kategori tertentu"""
    cap = cv2.VideoCapture(0)
    embeddings = load_embeddings()
    
    categories_display = ", ".join([c.upper() for c in categories])
    print(f"Area: {area.upper()}")
    print(f"Checking: {categories_display}")
    print("Press 'q' to exit")
    
    excluded = {"safety cone", "safety vest", "no safety vest", "machinery", "vehicle", "no safety cone"}
    recognized_name = None
    last_state = {}
    last_save_time = {}
    last_violation_state = {}
    last_passed_state = {}
    enabled_categories = set(categories)  # Track which categories are enabled
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Reset recognized_name for current frame
        recognized_name = None
        
        # Face detection & recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            recognized_name = recognize_face(frame, (x, y, w, h), embeddings)
            
            color = (0, 255, 0) if recognized_name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if recognized_name:
                cv2.putText(frame, f"✓ {recognized_name}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # YOLO APD detection for selected categories
        results = yolo_model(frame, conf=0.25)[0]
        
        # Draw bounding boxes with labels
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
            
            label = replace_label_names(label)
            norm = normalize_label(label)
            
            if norm in excluded:
                continue
            
            # Check hanya kategori yang sesuai area
            matched = False
            for category in categories:
                if category in label.lower():
                    matched = True
                    break
            
            if matched:
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
        
        # Save record for detected APD (with or without face recognition)
        # Get person name (recognized or "Unknown")
        person_name = recognized_name if recognized_name else "Unknown"
        
        # Re-run detection to get results for saving
        best_results = {}
        best_conf = {}
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
            
            label = replace_label_names(label)
            norm = normalize_label(label)
            
            if norm in excluded:
                continue
            
            # Check hanya kategori yang sesuai area
            for category in categories:
                if category in label.lower():
                    # Keep only highest confidence result per category
                    if category not in best_conf or conf > best_conf[category]:
                        best_results[category] = label
                        best_conf[category] = conf
        
        results_dict = {}
        for cat in ["mask", "glove", "helm", "glasses", "boots"]:
            # Only include if in selected categories for this area
            if cat in categories:
                if cat in best_results:
                    label = best_results[cat]
                    results_dict[f"is_{cat}"] = not label.lower().startswith("no")
                else:
                    results_dict[f"is_{cat}"] = False
            else:
                # If not in this area's categories, set to False (or you could exclude)
                results_dict[f"is_{cat}"] = False
        
        # Draw APD status on frame
        frame = draw_apd_status(frame, results_dict, enabled_categories)
        
        # Save record to Google Sheet
        save_check_record(person_name, results_dict, last_state, last_save_time)
        
        # Check and save violation if any APD is not compliant
        save_violation_record(frame, person_name, results_dict, enabled_categories, last_violation_state)
        
        # Check and save passed record if all APD are compliant
        save_passed_record(frame, person_name, results_dict, enabled_categories, last_passed_state)
        
        cv2.imshow("APD Check", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    clear_screen()

def save_check_record_single(name, category, result, last_state, last_save_time, debounce_seconds=2):
    """Save single category record to Google Sheet"""
    import time
    
    state_key = f"{name}_{category}"
    time_key = f"{state_key}_time"
    current_time = time.time()
    
    # Check if state is same as last time
    if result == last_state.get(state_key):
        return False
    
    # Check if enough time has passed since last save (debounce)
    last_save = last_save_time.get(time_key, 0)
    if current_time - last_save < debounce_seconds:
        return False
    
    # State changed AND debounce passed, convert to results_dict and save
    is_value = not result.lower().startswith("no")
    
    # For single category, set others to False
    results_dict_full = {
        "is_mask": is_value if category == "mask" else False,
        "is_glove": is_value if category == "glove" else False,
        "is_helm": is_value if category == "helm" else False,
        "is_glasses": is_value if category == "glasses" else False,
        "is_boots": is_value if category == "boots" else False
    }
    
    # Use save_check_record to save with full dict
    ret = save_check_record(name, results_dict_full, last_state, last_save_time, debounce_seconds=0)
    
    if ret:
        # Update last state and time
        last_state[state_key] = result
        last_save_time[time_key] = current_time
        print(f"✓ Record saved: {name} - {category.upper()}: {result}")
    
    return ret

def main_menu():
    """Main menu"""
    while True:
        clear_screen()
        print("\n" + "="*40)
        print("APD CHECK SYSTEM")
        print("="*40)
        print("1. Register Face")
        print("2. Check APD")
        print("3. View Records")
        print("4. Exit")
        choice = input("Pilih menu (1/2/3/4): ").strip()
        
        if choice == "1":
            clear_screen()
            name = input("Masukkan nama: ").strip()
            if name:
                register_face(name)
            clear_screen()
        elif choice == "2":
            check_apd()
        elif choice == "3":
            clear_screen()
            view_records()
            input("Tekan Enter untuk kembali ke menu utama...")
            clear_screen()
        elif choice == "4":
            clear_screen()
            print("Goodbye!")
            break
        else:
            print("❌ Pilihan tidak valid!")
            input("Tekan Enter untuk lanjut...")

def view_records():
    """View all records from Google Sheet"""
    worksheet = get_google_sheet()
    if worksheet is None:
        return
    
    try:
        # Get all values
        all_values = worksheet.get_all_values()
        
        if len(all_values) <= 1:  # Only header or empty
            print("Tidak ada records.")
            return
        
        # Parse data
        headers = all_values[0]
        records = all_values[1:]
        
        categories = ["mask", "glove", "helm", "glasses", "boots"]
        
        print("\n=== APD CHECK RECORDS ===")
        header_format = f"{'Name':<20}"
        for cat in categories:
            header_format += f" {cat.capitalize():<10}"
        header_format += f" {'Timestamp':<25}"
        print(f"\n{header_format}")
        print("-" * 110)
        
        for record in records:
            if len(record) >= 1:
                row_format = f"{record[0]:<20}"
                for i, cat in enumerate(categories):
                    col_idx = i + 1
                    if col_idx < len(record):
                        is_true = "✓" if record[col_idx].lower() == "true" else "✗"
                    else:
                        is_true = "-"
                    row_format += f" {is_true:<10}"
                
                # Timestamp
                if len(record) > len(categories) + 1:
                    timestamp = record[len(categories) + 1][:19]
                else:
                    timestamp = "-"
                row_format += f" {timestamp:<25}"
                print(row_format)
        
        # Summary statistics
        print("\n=== SUMMARY ===")
        total_records = len(records)
        print(f"Total Records: {total_records}")
        
        names = [r[0] for r in records if len(r) > 0]
        unique_persons = len(set(names))
        print(f"Total Persons: {unique_persons}")
        
        print("\nCompliance Rate:")
        for i, cat in enumerate(categories):
            col_idx = i + 1
            compliant = sum(1 for r in records if len(r) > col_idx and r[col_idx].lower() == "true")
            percentage = (compliant / total_records * 100) if total_records > 0 else 0
            print(f"  {cat.capitalize()}: {compliant}/{total_records} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error reading records: {e}")

if __name__ == "__main__":
    main_menu()