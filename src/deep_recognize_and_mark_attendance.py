import json
import os
import queue
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional

# Disable OpenCV OpenCL to prevent hangs/deadlocks 
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np
from deepface import DeepFace
import pyttsx3
import threading

# Global variables for async communication
recognition_queue = queue.Queue(maxsize=1)
results_lock = threading.Lock()
current_recognition_result = {
    "student_id": None,
    "label_text": "Scanning...",
    "is_marked": False
}

def speak_async(text: str):
    def run_speech():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run_speech, daemon=True).start()

from db import (
    get_connection,
    init_db,
    load_student_details,
    mark_attendance_if_needed,
    sync_students_from_csv,
)

MODEL_NAME = "Facenet512"
# Dynamic threshold for high-accuracy identification
DIST_THRESHOLD = 0.45 
MARGIN_REQUIREMENT = 0.08 # Must be at least 0.08 better than other candidates

# Voting Window for stability
WINDOW_SIZE = 5
REQUIRED_VOTES = 3

def get_cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return 1 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def load_embeddings() -> Dict[int, list]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT student_id, embedding FROM face_embeddings WHERE model_name = ?;",
        (MODEL_NAME,),
    )
    rows = cur.fetchall()
    conn.close()

    embeddings: Dict[int, list] = {}
    for sid, emb_json in rows:
        try:
            vec_list = json.loads(emb_json)
            # We now always expect a list of vectors from the new deep_train.py
            if isinstance(vec_list[0], (int, float)):
                embeddings[int(sid)] = [np.array(vec_list, dtype="float32")]
            else:
                embeddings[int(sid)] = [np.array(v, dtype="float32") for v in vec_list]
        except Exception:
            continue
    return embeddings

def recognition_worker(embeddings, student_details):
    """Background thread worker that performs heavy AI identification."""
    global current_recognition_result
    id_history = deque(maxlen=WINDOW_SIZE)
    marked_today = set()

    while True:
        try:
            face_roi = recognition_queue.get(block=True)
            if face_roi is None: break 

            face_resized = cv2.resize(face_roi, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            reps = DeepFace.represent(
                img_path=face_rgb,
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                align=True 
            )

            best_id = None
            best_dist = float("inf")
            second_best_dist = float("inf")
            
            if reps:
                vec = np.array(reps[0]["embedding"], dtype="float32")
                
                # 1. Compare against all embeddings for all students
                scores = []
                for sid, emb_list in embeddings.items():
                    # Find best match for this student
                    local_min = min([get_cosine_distance(e, vec) for e in emb_list])
                    scores.append((sid, local_min))
                
                # Sort by score
                scores.sort(key=lambda x: x[1])
                
                if scores:
                    best_id, best_dist = scores[0]
                    if len(scores) > 1:
                        second_best_dist = scores[1][1]

            # 2. Strict Identification: Threshold + Margin logic
            # Margin prevents misidentifying as a similar-looking student
            is_valid_match = (best_id is not None and 
                             best_dist < DIST_THRESHOLD and 
                             (second_best_dist - best_dist) > MARGIN_REQUIREMENT)

            if is_valid_match:
                id_history.append(best_id)
            else:
                id_history.append(None)

            # 3. Stability Voting
            votes = list(id_history)
            stable_id = None
            if votes.count(None) < (WINDOW_SIZE - REQUIRED_VOTES):
                valid_votes = [v for v in votes if v is not None]
                if valid_votes:
                    most_common = max(set(valid_votes), key=valid_votes.count)
                    if valid_votes.count(most_common) >= REQUIRED_VOTES:
                        stable_id = most_common

            # 4. Update Global Result
            with results_lock:
                if stable_id in student_details:
                    d = student_details[stable_id]
                    if stable_id not in marked_today:
                        created, _ = mark_attendance_if_needed(stable_id)
                        if created:
                            speak_async(f"Attendance marked for {d['name']}")
                            marked_today.add(stable_id)
                    
                    current_recognition_result = {
                        "student_id": stable_id,
                        "label_text": f"{d['name']}_{d['roll_no']}_{d['branch']}" + ("-MARKED" if stable_id in marked_today else ""),
                        "confidence_score": best_dist,
                        "is_marked": stable_id in marked_today
                    }
                else:
                    # If we found a match but it failed voting or margin, call it scanning
                    # If best_dist is very far, it's truly an Unknown student
                    is_far = best_dist > 0.6 if best_id else True
                    current_recognition_result = {
                        "student_id": None,
                        "label_text": "Scanning..." if not is_far else "Unknown!",
                        "confidence_score": best_dist if best_id else 1.0,
                        "is_marked": False
                    }
            
            recognition_queue.task_done()
        except Exception:
            time.sleep(0.1)

def main() -> None:
    init_db()
    sync_students_from_csv()
    student_details = load_student_details()
    embeddings = load_embeddings()

    if not embeddings:
        print("No deep embeddings found. Please train the model first.")
        return

    worker_thread = threading.Thread(target=recognition_worker, args=(embeddings, student_details), daemon=True)
    worker_thread.start()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

        for i, (x, y, w, h) in enumerate(faces):
            with results_lock:
                status = current_recognition_result
            
            color = (0, 255, 0) if status["is_marked"] else (0, 255, 255) if status["student_id"] else (0, 0, 255)
            label = status["label_text"] if i == 0 else "" 
            
            if i == 0 and recognition_queue.empty():
                pad = 30
                face_roi = frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
                if face_roi.size > 0:
                    try:
                        recognition_queue.put_nowait(face_roi.copy())
                    except queue.Full: pass

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Advanced AI Attendance (Async) - Press Q to Exit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    recognition_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
