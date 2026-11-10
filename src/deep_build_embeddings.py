import json
import os

# Disable OpenCV OpenCL to prevent hangs/deadlocks 
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

from collections import defaultdict
from typing import Dict, List

import numpy as np
from deepface import DeepFace

from db import get_connection, init_db, sync_students_from_csv


DATASET_DIR = os.path.join("data", "dataset")
MODEL_NAME = "VGG-Face"


def build_embeddings() -> None:
    if not os.path.isdir(DATASET_DIR):
        print("Dataset directory not found:", DATASET_DIR)
        return

    # student_id -> list of embedding vectors
    per_student: Dict[int, List[np.ndarray]] = defaultdict(list)

    files = [
        f
        for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(".jpg") and f.startswith("user.")
    ]
    if not files:
        print("No images found in dataset. Run capture_faces.py first.")
        return

    print(f"Found {len(files)} images. Extracting {MODEL_NAME} embeddings...")

    for filename in files:
        parts = filename.split(".")
        if len(parts) < 3:
            continue
        try:
            student_id = int(parts[1])
        except ValueError:
            continue

        img_path = os.path.join(DATASET_DIR, filename)
        try:
            reps = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False,
            )
        except Exception as exc:
            print(f"Skipping {img_path}: {exc}")
            continue

        if not reps:
            continue

        emb = np.array(reps[0]["embedding"], dtype="float32")
        per_student[student_id].append(emb)

    if not per_student:
        print("No embeddings could be extracted.")
        return

    conn = get_connection()
    cur = conn.cursor()

    # Remove old embeddings for this model.
    cur.execute(
        "DELETE FROM face_embeddings WHERE model_name = ?;",
        (MODEL_NAME,),
    )

    for student_id, vectors in per_student.items():
        # Instead of just the mean, let's store a representative sample (e.g. 20 images)
        # across the sequence to capture different angles/lighting.
        sample_size = min(20, len(vectors))
        indices = np.linspace(0, len(vectors) - 1, sample_size, dtype=int)
        sampled_vectors = [vectors[i].tolist() for i in indices]
        
        emb_json = json.dumps(sampled_vectors)

        cur.execute(
            """
            INSERT INTO face_embeddings (student_id, model_name, embedding)
            VALUES (?, ?, ?);
            """,
            (student_id, MODEL_NAME, emb_json),
        )

    conn.commit()
    conn.close()
    print(f"Saved embeddings for {len(per_student)} students using {MODEL_NAME}.")


def main() -> None:
    init_db()
    sync_students_from_csv()
    build_embeddings()


if __name__ == "__main__":
    main()

