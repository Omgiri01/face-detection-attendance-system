import csv
import os
import cv2
import shutil
from preprocessing import preprocess_face
from db import sync_students_from_csv

DATASET_DIR = os.path.join("data", "dataset")
STUDENTS_CSV = os.path.join("data", "students.csv")
UPLOADS_DIR = "uploads"
# Hardcode to exactly what the system is piping in
BULK_CSV = "bulk_enroll.csv"

import gdown
import re

def download_from_drive(url, output_path):
    """Extracts File ID from Google Drive URL and downloads using gdown."""
    try:
        # Regex to find the ID in various Drive URL formats
        match = re.search(r'id=([a-zA-Z0-9_-]+)', url) or re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if not match:
            print(f"Could not extract ID from URL: {url}")
            return False
        
        file_id = match.group(1)
        import requests
        download_url = f"https://drive.google.com/uc?id={file_id}"
        r = requests.get(download_url, stream=True)
        if r.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.getsize(output_path) > 5000:
                return True
            os.remove(output_path)
        return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def bulk_import():
    """
    Reads bulk_enroll.csv and processes images.
    If 'Upload your photos' contains URLs, downloads them first.
    """
    if not os.path.exists(BULK_CSV):
        print(f"Error: {BULK_CSV} not found.")
        return False
        
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    success_count = 0
    with open(BULK_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = {k.lower().strip(): k for k in reader.fieldnames}
        
        name_col = fieldnames.get('name') or fieldnames.get('full name')
        id_col = fieldnames.get('roll number') or fieldnames.get('id')
        branch_col = fieldnames.get('branch')
        for row in reader:
            name = ""
            if name_col: name = row.get(name_col, "").strip()
            sid = ""
            if id_col: sid = row.get(id_col, "").strip()
            branch = row.get(branch_col, "").strip() if branch_col else ""
            
            # dynamically grab all photo columns
            photo_data_list = []
            for k, v in row.items():
                if k and 'photo' in k.lower():
                    if v.strip(): photo_data_list.append(v.strip())
            
            # also add the old single column format if found
            photo_col = fieldnames.get('upload your photos (3-5 images)') or fieldnames.get('photo')
            if photo_col and row.get(photo_col, None):
                photo_data_list.append(row[photo_col].strip())

            if not name or not sid:
                continue

            print(f"--- Processing: {name} (ID: {sid}) ---")
            
            # Step 1: Handle Downloads if links exist
            photo_links = []
            for pd in photo_data_list:
                photo_links.extend([link.strip().strip('"') for link in pd.split(',')])
            local_files = []
            
            for idx, link in enumerate(photo_links):
                if 'drive.google.com' in link:
                    local_path = os.path.join(UPLOADS_DIR, f"{sid}_{idx}.jpg")
                    if not os.path.exists(local_path):
                        print(f"Downloading photo {idx+1} for {name}...")
                        if download_from_drive(link, local_path):
                            local_files.append(local_path)
                    else:
                        local_files.append(local_path)
                else:
                    # Treat as local filename
                    local_path = os.path.join(UPLOADS_DIR, link)
                    if os.path.exists(local_path):
                        local_files.append(local_path)

            if not local_files:
                print(f"No photos available for {name}.")
                continue

            # Step 2: Face Extraction
            face_count = 0
            for photo_path in local_files:
                img = cv2.imread(photo_path)
                if img is None: continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                for (x, y, w, h) in faces:
                    face_count += 1
                    face_img = gray[y:y+h, x:x+w]
                    face_img = preprocess_face(face_img)
                    
                    if face_count <= 20: # Limit samples from diverse photos
                        out_name = f"user.{sid}.{face_count}.jpg"
                        cv2.imwrite(os.path.join(DATASET_DIR, out_name), face_img)
            
            if face_count > 0:
                add_to_main_csv(sid, name, branch)
                success_count += 1
                print(f"Imported {face_count} face samples for {name}")

    sync_students_from_csv()
    return success_count > 0

def add_to_main_csv(sid, name, branch):
    # Ensure students.csv exists with headers
    headers = ["id", "name", "roll_no", "branch"]
    if not os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
            
    rows = []
    with open(STUDENTS_CSV, 'r', newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    
    ids = [r[0] for r in rows[1:]]
    if str(sid) not in ids:
        with open(STUDENTS_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([sid, name, sid, branch])

if __name__ == "__main__":
    bulk_import()
