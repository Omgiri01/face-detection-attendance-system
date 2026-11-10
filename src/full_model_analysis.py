"""
Full model analysis and evaluation.
- Train/test split (80/20) for honest accuracy
- Per-student accuracy and confusion matrix
- Writes report to data/evaluation_report.txt
"""
import os
import random
from collections import defaultdict

import cv2
import numpy as np

from preprocessing import preprocess_face


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_report.txt")


def load_dataset() -> tuple[list[tuple[np.ndarray, int]], dict[int, int]]:
    """Load (image, label) pairs. Returns samples and count per user."""
    samples: list[tuple[np.ndarray, int]] = []
    count_per_user: dict[int, int] = defaultdict(int)

    for filename in sorted(os.listdir(DATASET_DIR)):
        if not filename.lower().endswith(".jpg") or not filename.startswith("user."):
            continue
        parts = filename.split(".")
        if len(parts) < 3:
            continue
        try:
            uid = int(parts[1])
        except ValueError:
            continue

        path = os.path.join(DATASET_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = preprocess_face(img)
        samples.append((img, uid))
        count_per_user[uid] += 1

    return samples, dict(count_per_user)


def split_train_test(samples: list[tuple[np.ndarray, int]], test_frac: float = 0.2, seed: int = 42):
    """Split per-user so each user has train and test images."""
    random.seed(seed)
    by_user: dict[int, list[tuple[np.ndarray, int]]] = defaultdict(list)
    for img, uid in samples:
        by_user[uid].append((img, uid))

    train_list: list[tuple[np.ndarray, int]] = []
    test_list: list[tuple[np.ndarray, int]] = []

    for uid, lst in by_user.items():
        random.shuffle(lst)
        n_test = max(1, int(len(lst) * test_frac))
        n_train = len(lst) - n_test
        train_list.extend(lst[:n_train])
        test_list.extend(lst[n_train:])

    return train_list, test_list


def main() -> None:
    print("Loading dataset...")
    samples, count_per_user = load_dataset()
    if not samples:
        print("No images found.")
        return

    print(f"Total images: {len(samples)}, Users: {len(count_per_user)}")
    for uid, cnt in sorted(count_per_user.items()):
        print(f"  User {uid}: {cnt} images")

    train_list, test_list = split_train_test(samples)
    print(f"\nTrain: {len(train_list)}, Test: {len(test_list)}")

    print("Training LBPH on train split...")
    train_faces = [x[0] for x in train_list]
    train_ids = np.array([x[1] for x in train_list])
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_faces, train_ids)

    print("Evaluating on held-out test set...")
    correct = 0
    total = 0
    per_user_correct: dict[int, int] = defaultdict(int)
    per_user_total: dict[int, int] = defaultdict(int)
    confusion: dict[tuple[int, int], int] = defaultdict(int)

    for img, true_id in test_list:
        pred_id, conf = recognizer.predict(img)
        total += 1
        per_user_total[true_id] += 1
        if pred_id == true_id:
            correct += 1
            per_user_correct[true_id] += 1
        confusion[(true_id, pred_id)] += 1

    overall_acc = (correct / total) * 100.0 if total else 0

    lines = [
        "=" * 60,
        "FACE DETECTION ATTENDANCE SYSTEM - MODEL EVALUATION REPORT",
        "=" * 60,
        "",
        "Dataset summary",
        "-" * 40,
        f"Total images: {len(samples)}",
        f"Unique users (students): {len(count_per_user)}",
        f"Images per user: {dict(sorted(count_per_user.items()))}",
        "",
        "Split (80% train / 20% test)",
        "-" * 40,
        f"Train images: {len(train_list)}",
        f"Test images: {len(test_list)}",
        "",
        "Overall accuracy (on held-out test set)",
        "-" * 40,
        f"Correct: {correct} / {total}",
        f"Accuracy: {overall_acc:.2f}%",
        "",
        "Per-student accuracy",
        "-" * 40,
    ]

    for uid in sorted(per_user_total.keys()):
        c = per_user_correct[uid]
        t = per_user_total[uid]
        acc = (c / t) * 100.0 if t else 0
        lines.append(f"  User {uid}: {c}/{t} = {acc:.2f}%")

    lines.extend([
        "",
        "Confusion matrix (true_id -> pred_id, count)",
        "-" * 40,
    ])
    for (true_id, pred_id), cnt in sorted(confusion.items()):
        if true_id != pred_id:
            lines.append(f"  {true_id} -> {pred_id}: {cnt} (misclassification)")

    if not any("misclassification" in l for l in lines):
        lines.append("  No misclassifications.")

    lines.extend([
        "",
        "Limitations & notes",
        "-" * 40,
        "- LBPH is robust to lighting but sensitive to pose/angle changes.",
        "- User 4 has fewer images (55) vs others (240) — may affect performance.",
        "- Live attendance uses temporal smoothing (window=5, required=3) to reduce false marks.",
        "- This report uses held-out test data; in-training accuracy would be higher.",
        "",
        "=" * 60,
    ])

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
