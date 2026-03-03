# APD Detection System

A professional AI-based Personal Protective Equipment (PPE) detection system using YOLOv11 and Streamlit. This system monitors compliance in real-time, processes images/videos, and logs violations.

---

### 🚀 Features

- **Live Monitoring**: Real-time CCTV stream analysis with optimized frame-skipping.
- **Image Analysis**: Deep analysis for uploaded photos with detailed compliance reports.
- **Video Analytics**: Automated frame-by-frame (optimized) processing for recorded footage.
- **Security Records**: Automated logging of violations and passes to local storage and Google Sheets.
- **Dynamic Configuration**: Adjust detection threshold, area-specific rules, and PPE categories on the fly.

---

### 🛠️ Tech Stack

- **Core**: Python 3.10+
- **AI Model**: Ultralytics YOLOv11
- **UI Framework**: Streamlit
- **Computer Vision**: OpenCV, Pillow
- **Database/Logging**: Google Sheets API, JSON local storage

---

### 📥 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Credentials**
   - Place your Google Sheets `credentials.json` in the root directory if you want to use the logging feature.

---

### 💻 Usage

Run the application using Streamlit:

```bash
streamlit run streamlit_app.py
```

---

### 📂 Project Structure

- `streamlit_app.py`: Main application entry point.
- `model/`: Contains trained YOLO weights (`best.pt`).
- `violations/`: Directory for recorded safety violations (Git-ignored).
- `passed/`: Directory for compliant detection records (Git-ignored).
- `config.json`: Area-specific configuration settings.

---

### ⚙️ Performance Optimization

The system includes a **Detection Frequency** setting located in the sidebar. 
- Lower values (e.g., 1) process every frame for maximum accuracy.
- Higher values (e.g., 5-20) skip frames to significantly improve performance on lower-end hardware or long video streams.

---

*(Developed for Advanced Safety Monitoring)*
