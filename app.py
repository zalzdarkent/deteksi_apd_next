import cv2
import json
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np

# Create necessary directories
DATA_DIR = Path("data")
FACES_DIR = DATA_DIR / "faces"
RECORDS_DIR = DATA_DIR / "records"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"

DATA_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
RECORDS_DIR.mkdir(exist_ok=True)

# YOLO Models
yolo_model = YOLO('model/yolo_kaggle/best.pt')

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

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
    """Tentukan warna label berdasarkan tipe (No- = merah, lainnya = hijau)"""
    if label.startswith("NO-") or label.startswith("No-"):
        return (0, 0, 255)  # Red
    else:
        return (0, 255, 0)   # Green

def save_check_record(name, results_dict, last_state, last_save_time, debounce_seconds=2):
    """Save APD check result to JSON only if ANY state changed AND debounce time passed"""
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
    
    # State changed AND debounce passed, save it
    record = {
        "name": name,
        "is_mask": results_dict.get("is_mask", False),
        "is_glove": results_dict.get("is_glove", False),
        "is_helm": results_dict.get("is_helm", False),
        "timestamp": datetime.now().isoformat()
    }
    
    records_file = RECORDS_DIR / f"{name}_records.json"
    records = []
    
    if records_file.exists():
        with open(records_file, 'r') as f:
            records = json.load(f)
    
    records.append(record)
    
    with open(records_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    # Update last state and time
    last_state[state_key] = current_state
    last_save_time[time_key] = current_time
    
    print(f"✓ Record saved: {name} - Mask: {record['is_mask']}, Glove: {record['is_glove']}, Helm: {record['is_helm']}")
    return True

def check_apd():
    """Main check APD - pilih kategori atau all"""
    while True:
        clear_screen()
        print("\n" + "="*40)
        print("PILIH KATEGORI APD")
        print("="*40)
        print("1. Masker (Mask/No-Mask)")
        print("2. Sarung Tangan (Glove/No-Glove)")
        print("3. Helm (Helm/No-Helm)")
        print("4. Check ALL (Mask, Glove, Helm)")
        print("5. Kembali ke Menu Utama")
        choice = input("Pilih (1/2/3/4/5): ").strip()
        
        if choice == "1":
            clear_screen()
            check_apd_category("mask")
            break
        elif choice == "2":
            clear_screen()
            check_apd_category("glove")
            break
        elif choice == "3":
            clear_screen()
            check_apd_category("helm")
            break
        elif choice == "4":
            clear_screen()
            check_apd_all()
            break
        elif choice == "5":
            clear_screen()
            return
        else:
            print("❌ Pilihan tidak valid!")
            input("Tekan Enter untuk lanjut...")

def check_apd_category(category):
    """Check APD untuk satu kategori saja"""
    cap = cv2.VideoCapture(0)
    embeddings = load_embeddings()
    
    if not embeddings:
        print("❌ Tidak ada face yang terdaftar! Silakan register dulu.")
        cap.release()
        input("Tekan Enter untuk kembali...")
        return
    
    print(f"Checking: {category.upper()}")
    print("Press 'q' to exit")
    
    excluded = {"safety cone", "safety vest", "no safety vest", "machinery", "vehicle", "no safety cone"}
    recognized_name = None
    last_state = {}
    last_save_time = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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
        
        # YOLO APD detection
        results = yolo_model(frame, conf=0.1)[0]
        
        best_result = None
        best_conf = 0
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
            
            label = replace_label_names(label)
            norm = normalize_label(label)
            
            if norm in excluded:
                continue
            
            if category in label.lower():
                if conf > best_conf:
                    best_result = label
                    best_conf = conf
        
        # Draw result
        if best_result:
            color = get_label_color(best_result)
            display_text = f"{category.upper()}: {best_result} {best_conf:.2f}"
            cv2.putText(frame, display_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save record
            if recognized_name:
                results_dict = {f"is_{category}": not best_result.lower().startswith("no")}
                save_check_record_single(recognized_name, category, best_result, last_state, last_save_time)
        else:
            cv2.putText(frame, f"{category.upper()}: -", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        cv2.imshow("APD Check", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    clear_screen()

def check_apd_all():
    """Check APD - cek SEMUA kategori sekaligus"""
    cap = cv2.VideoCapture(0)
    embeddings = load_embeddings()
    
    if not embeddings:
        print("❌ Tidak ada face yang terdaftar! Silakan register dulu.")
        cap.release()
        input("Tekan Enter untuk kembali...")
        return
    
    print(f"Checking: MASK, GLOVE, HELM")
    print("Press 'q' to exit")
    
    excluded = {"safety cone", "safety vest", "no safety vest", "machinery", "vehicle", "no safety cone"}
    recognized_name = None
    last_state = {}
    last_save_time = {}
    
    categories = ["mask", "glove", "helm"]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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
        
        # YOLO APD detection for ALL categories
        results = yolo_model(frame, conf=0.1)[0]
        
        # Track best result per category (highest confidence)
        best_results = {}  # {category: label}
        best_conf = {}     # {category: confidence}
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
            
            label = replace_label_names(label)
            norm = normalize_label(label)
            
            if norm in excluded:
                continue
            
            # Check semua kategori
            for category in categories:
                if category in label.lower():
                    # Keep only highest confidence result per category
                    if category not in best_conf or conf > best_conf[category]:
                        best_results[category] = label
                        best_conf[category] = conf
        
        # Draw results untuk semua kategori
        y_offset = 30
        for category in categories:
            if category in best_results:
                label = best_results[category]
                color = get_label_color(label)
                conf = best_conf[category]
                display_text = f"{category.upper()}: {label} {conf:.2f}"
            else:
                color = (100, 100, 100)
                display_text = f"{category.upper()}: -"
            
            cv2.putText(frame, display_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Save record only if person recognized (save ALL results at once)
        if recognized_name:
            results_dict = {}
            for category in categories:
                # Check if kategori ini ada hasil dan tidak "No-"
                if category in best_results:
                    label = best_results[category]
                    results_dict[f"is_{category}"] = not label.lower().startswith("no")
                else:
                    results_dict[f"is_{category}"] = False
            
            save_check_record(recognized_name, results_dict, last_state, last_save_time)
        
        cv2.imshow("APD Check", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    clear_screen()

def save_check_record_single(name, category, result, last_state, last_save_time, debounce_seconds=2):
    """Save single category record"""
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
    
    # State changed AND debounce passed, save it
    record = {
        "name": name,
        "category": category,
        "result": result,
        "timestamp": datetime.now().isoformat()
    }
    
    records_file = RECORDS_DIR / f"{name}_records.json"
    records = []
    
    if records_file.exists():
        with open(records_file, 'r') as f:
            records = json.load(f)
    
    records.append(record)
    
    with open(records_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    # Update last state and time
    last_state[state_key] = result
    last_save_time[time_key] = current_time
    
    print(f"✓ Record saved: {name} - {category.upper()}: {result}")
    return True

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
    """View all records"""
    records_files = list(RECORDS_DIR.glob("*.json"))
    if not records_files:
        print("Tidak ada records.")
        return
    
    print("\n=== APD CHECK RECORDS ===")
    for record_file in records_files:
        with open(record_file, 'r') as f:
            records = json.load(f)
            name = record_file.stem.replace("_records", "")
            print(f"\n{name}:")
            for record in records[-10:]:  # Show last 10 records
                mask_status = "✓ Mask" if record.get('is_mask', False) else "✗ No-Mask"
                glove_status = "✓ Glove" if record.get('is_glove', False) else "✗ No-Glove"
                helm_status = "✓ Helm" if record.get('is_helm', False) else "✗ No-Helm"
                print(f"  [{record['timestamp']}]")
                print(f"    {mask_status} | {glove_status} | {helm_status}")

if __name__ == "__main__":
    main_menu()