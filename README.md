# Face Detection & Automated Biometric Attendance System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-Deep%20Embeddings-red.svg)](https://github.com/serengil/deepface)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg?logo=sqlite)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, privacy-focused computer vision attendance monitoring system combining real-time face detection, anti-spoofing liveness verification, deep facial embeddings, and local database management.

---

## 📌 Architecture & System Flow

```
[Camera Stream / Bulk Uploads]
             │
             ▼
[Face Detection & Landmark Localization]
             │
             ▼
[Anti-Spoofing / Liveness Verification]
             │
             ├── If Spoof ──> [Reject & Alert]
             │
             ▼ (Real Face)
[Feature Extraction & Classification]
  (LBPH Local Binary Pattern / DeepFace Cosine Embeddings)
             │
             ▼
[Identity Match & Threshold Verification]
             │
             ▼
[Automated Attendance Logging]
  (SQLite DB Records + Daily CSV Exports + Text-to-Speech Confirmation)
```

---

## ✨ Features

- **Dual Recognition Pipelines:**
  - **Fast Classical Pipeline:** Local Binary Patterns Histograms (LBPH) via OpenCV for lightweight, real-time edge processing.
  - **Deep Learning Pipeline:** DeepFace facial representation with high-dimensional embeddings for challenging lighting and pose angles.
- **Anti-Spoofing & Liveness Detection (`liveness.py`):**
  - Defends against photograph/screen presentation attacks using motion heuristics and texture analysis.
- **Data Ingestion & Bulk Enrollment (`bulk_import.py`, `capture_faces.py`):**
  - Live webcam face capture with auto-crop and normalization.
  - Bulk import pipeline supporting directory imports with automatic augmentation (`augment_dataset.py`).
- **Database & Reporting (`db.py`, `export_attendance.py`):**
  - Embedded SQLite database storing student metadata, registration timestamps, and attendance logs.
  - Export capabilities to dated CSV reports (`data/attendance/`).
- **Audio Feedback (`pyttsx3`):**
  - Real-time voice announcements confirming student attendance upon recognition.
- **Interactive Terminal Menu (`main_menu.py`):**
  - Unified CLI control center to capture faces, train models, launch attendance tracking, export data, and inspect logs.

---

## 🔒 Privacy & Data Protection Notice

In compliance with data privacy best practices:
- **No real biometric image datasets or identifiable student records are included in this repository.**
- Pre-trained multi-gigabyte production weights (`data/models/trainer.yml`) and student databases are excluded via `.gitignore`.
- An example metadata schema is provided in `data/students.csv.example`.
- You can populate your own dataset using the included enrollment tools.

---

## 📂 Repository Structure

```text
face-detection-attendance-system/
├── src/
│   ├── main_menu.py                          # Unified CLI entrypoint
│   ├── capture_faces.py                      # Interactive webcam face capture & normalization
│   ├── bulk_import.py                        # Batch import from image directories
│   ├── augment_dataset.py                    # Dataset augmentation (contrast, flips, rotations)
│   ├── liveness.py                           # Anti-spoofing and liveness validation
│   ├── train_model.py                        # LBPH model trainer
│   ├── recognize_and_mark_attendance.py      # Real-time LBPH attendance tracking
│   ├── deep_build_embeddings.py              # Generates deep neural network face representations
│   ├── deep_train.py                         # Deep learning classification model trainer
│   ├── deep_recognize_and_mark_attendance.py # Deep learning real-time recognition loop
│   ├── db.py                                 # SQLite schema creation and query interface
│   ├── export_attendance.py                  # Generates formatted attendance sheets
│   ├── evaluate_model.py                     # Accuracy, ROC, and confusion matrix evaluator
│   ├── full_model_analysis.py                # Comprehensive performance analytics
│   ├── camera_test.py                        # Video device diagnostics utility
│   ├── clear_attendance.py                   # Reset or purge attendance sessions
│   ├── preprocessing.py                      # Histogram equalization and alignment helpers
│   └── show_db.py                            # Inspect registered users and attendance
├── data/
│   ├── students.csv.example                  # Template schema for student enrollment
│   ├── dataset/                              # Clean dataset folder (populated during capture)
│   ├── dataset_outliers/                     # Filtered outliers storage
│   ├── models/                               # Generated model weights directory
│   └── attendance/                           # Generated attendance logs
├── uploads/                                  # Staging directory for image imports
├── figures/                                  # Architectural and workflow diagrams
├── requirements.txt                          # Python library dependencies
├── .gitignore                                # Git rules protecting private biometric data
├── LICENSE                                   # MIT License
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Omgiri01/face-detection-attendance-system.git
   cd face-detection-attendance-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Quickstart Workflow

Run the main dashboard menu:
```bash
python src/main_menu.py
```

Or execute individual steps via script:

1. **Test Camera:**
   ```bash
   python src/camera_test.py
   ```

2. **Enroll a Student:**
   ```bash
   python src/capture_faces.py
   ```
   Enter the student's ID and name. The camera captures and crops 60–100 normalized facial samples into `data/dataset/`.

3. **Train Recognition Model:**
   - For fast LBPH training:
     ```bash
     python src/train_model.py
     ```
   - For DeepFace neural embeddings:
     ```bash
     python src/deep_build_embeddings.py
     python src/deep_train.py
     ```

4. **Launch Attendance Monitoring:**
   ```bash
   python src/recognize_and_mark_attendance.py
   ```
   When a recognized face passes liveness checks, attendance is recorded in SQLite and verified via text-to-speech. Press `Q` to exit.

5. **Export Attendance Report:**
   ```bash
   python src/export_attendance.py
   ```
   Generates a structured CSV in `data/attendance/attendance_YYYY-MM-DD.csv`.

---

## 🛠 Tech Stack

- **Computer Vision:** OpenCV (`cv2`), DeepFace
- **Numerical Processing:** NumPy
- **Database:** SQLite3
- **Audio Output:** `pyttsx3`
- **Progress & Networking:** `tqdm`, `requests`, `gdown`

---

## 📄 Research Paper Publication

This repository contains the official implementation of the research paper:
> **A Hybrid Approach to Automated Face Detection Attendance System: Integrating FaceNet512 Deep Embeddings with LBPH Fallback Mechanisms**  
> *Authors:* **Om Giri**, **Sujeet Sharma**, et al.  
> *Affiliation:* Department of Mechanical Engineering, National Institute of Technology Raipur (NIT Raipur)  
> 📑 **Complete Paper:** [`RCP_06.pdf`](RCP_06.pdf)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

