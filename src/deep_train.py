import os
import json
import sqlite3
import numpy as np
import cv2
from deepface import DeepFace
from tqdm import tqdm

DATASET_DIR = os.path.join("data", "dataset")
DB_PATH = os.path.join("data", "attendance.db")
MODEL_NAME = "Facenet512" 

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def init_embedding_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS face_embeddings;") 
    cur.execute("""
        CREATE TABLE face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            embedding TEXT,
            model_name TEXT
        )
    """)
    conn.commit()
    conn.close()

def main(status_callback=None):
    if not os.path.isdir(DATASET_DIR): return

    if status_callback: status_callback("Initializing Ultra-Deep Calibration...")
    init_embedding_table()
    
    student_images = {}
    for filename in os.listdir(DATASET_DIR):
        if filename.lower().endswith(".jpg"):
            parts = filename.split(".")
            if len(parts) >= 3 and parts[0] == "user":
                sid = parts[1]
                if sid not in student_images:
                    student_images[sid] = []
                if "_" not in filename:
                    student_images[sid].insert(0, filename)
                elif len(student_images[sid]) < 100: # Increase pool for diversity filtering
                    student_images[sid].append(filename)

    if status_callback: status_callback(f"Extracting Ultra-Dense profiles for {len(student_images)} students...")
    
    conn = get_db_connection()
    cur = conn.cursor()

    for i, (sid, images) in enumerate(student_images.items()):
        if status_callback: status_callback(f"Mapping Student {sid} (Deep Profile)...")
        
        diverse_embeddings = []
        for img_name in images:
            img_path = os.path.join(DATASET_DIR, img_name)
            try:
                # Use faster backend for extraction, enforce detection to ensure quality
                res = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=True, detector_backend="opencv")
                if res:
                    new_emb = res[0]["embedding"]
                    if not diverse_embeddings:
                        diverse_embeddings.append(new_emb)
                    else:
                        # Check diversity: we want unique angles/lighting
                        min_dist = min([1 - (np.dot(new_emb, e) / (np.linalg.norm(new_emb) * np.linalg.norm(e))) for e in diverse_embeddings])
                        if min_dist > 0.04: # Slightly lower threshold to pack more detail
                            diverse_embeddings.append(new_emb)
                    
                    if len(diverse_embeddings) >= 32: break # ULTRA-DENSE PROFILE
            except Exception:
                continue
        
        if diverse_embeddings:
            cur.execute("""
                INSERT INTO face_embeddings (student_id, embedding, model_name)
                VALUES (?, ?, ?)
            """, (sid, json.dumps(diverse_embeddings), MODEL_NAME))
            conn.commit()

    conn.close()
    if status_callback: status_callback("Ultra-Deep Model Calibration Complete.")

if __name__ == "__main__":
    main()
