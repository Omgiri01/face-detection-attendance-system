import os
from typing import List, Tuple

import cv2
import numpy as np

from preprocessing import preprocess_face


DATASET_DIR = os.path.join("data", "dataset")
MODELS_DIR = os.path.join("data", "models")


os.makedirs(MODELS_DIR, exist_ok=True)


def get_images_and_labels() -> Tuple[List[np.ndarray], np.ndarray]:
    faces: List[np.ndarray] = []
    ids = []

    if not os.path.isdir(DATASET_DIR):
        return faces, np.array(ids)

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue

        path = os.path.join(DATASET_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        try:
            # Expected filename format: user.<id>.<count>.jpg
            parts = filename.split(".")
            id_ = int(parts[1])
        except Exception:
            continue

        img = preprocess_face(img)
        faces.append(img)
        ids.append(id_)

    return faces, np.array(ids)


def main(status_callback=None) -> None:
    if status_callback: status_callback("Preparing images for LBPH training...")
    faces, ids = get_images_and_labels()
    if len(faces) == 0:
        if status_callback: status_callback("No training images found.")
        print("No training images found. Run capture_faces.py first.")
        return

    if status_callback: status_callback(f"Training LBPH model on {len(faces)} samples...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)

    model_path = os.path.join(MODELS_DIR, "trainer.yml")
    recognizer.write(model_path)
    
    if status_callback: status_callback("LBPH Training Complete.")
    print(f"Model trained on {len(faces)} images, saved to {model_path}")

if __name__ == "__main__":
    main()

