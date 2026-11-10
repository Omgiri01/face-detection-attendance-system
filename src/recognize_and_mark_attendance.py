import os
from collections import defaultdict, deque
from datetime import datetime, timedelta

import cv2
import pyttsx3
import threading

def speak_async(text: str):
    def run_speech():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

from db import (
    init_db,
    load_students_from_db,
    load_student_details,
    mark_attendance_if_needed,
    sync_students_from_csv,
)
from preprocessing import preprocess_face

MODELS_DIR = os.path.join("data", "models")
CONFIDENCE_THRESHOLD = 80.0
WINDOW_SIZE = 5
REQUIRED_COUNT = 3
DATA_DIR = "data"

def main() -> None:
    init_db()
    sync_students_from_csv()
    student_details = load_student_details()

    model_path = os.path.join(MODELS_DIR, "trainer.yml")
    if not os.path.exists(model_path):
        print("Model not found. Run train_model.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Fast Mode (LBPH) Running. Press 'q' to quit.")

    history: dict[int, deque[int]] = defaultdict(
        lambda: deque(maxlen=WINDOW_SIZE)
    )
    marked_today = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            roi = gray[y : y + h, x : x + w]
            roi = preprocess_face(roi)

            label = "Unknown"
            color = (0, 0, 255)

            id_pred, confidence = recognizer.predict(roi)

            if confidence < CONFIDENCE_THRESHOLD and id_pred in student_details:
                history[id_pred].append(1)
            else:
                history[id_pred].append(0)

            if (
                id_pred in student_details
                and sum(history[id_pred]) >= REQUIRED_COUNT
                and confidence < CONFIDENCE_THRESHOLD
            ):
                details = student_details[id_pred]
                name = details["name"]
                roll = details["roll_no"]
                branch = details["branch"]
                
                label = f"{name}_{roll}_{branch}"
                color = (0, 255, 0)

                if id_pred not in marked_today:
                    created, _ = mark_attendance_if_needed(id_pred)
                    if created:
                        msg = f"Marked attendance for {name}"
                        speak_async(msg)
                        marked_today.add(id_pred)
                
                if id_pred in marked_today:
                    label += "-MARKED"

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Attendance (LBPH) - q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
