import os

import cv2
import numpy as np


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "trainer.yml")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_model.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    if not os.path.isdir(DATASET_DIR):
        print("Dataset folder not found at", DATASET_DIR)
        return

    total = 0
    correct = 0

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue
        if not filename.startswith("user."):
            continue

        parts = filename.split(".")
        if len(parts) < 3:
            continue

        try:
            true_id = int(parts[1])
        except ValueError:
            continue

        path = os.path.join(DATASET_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (200, 200))
        pred_id, conf = recognizer.predict(img)

        total += 1
        if pred_id == true_id:
            correct += 1

    if total == 0:
        print("No test images found in", DATASET_DIR)
        return

    acc = (correct / total) * 100.0
    print(f"Tested on {total} images.")
    print(f"Correct predictions: {correct}")
    print(f"Approximate dataset accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()


