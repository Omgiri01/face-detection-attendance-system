import csv
import os

import cv2

from preprocessing import preprocess_face


DATASET_DIR = os.path.join("data", "dataset")
STUDENTS_CSV = os.path.join("data", "students.csv")


os.makedirs(DATASET_DIR, exist_ok=True)


if not os.path.exists(STUDENTS_CSV):
    with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name"])


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def add_student_to_csv(student_id: int, name: str) -> None:
    rows = []
    if os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

    ids = {row[0] for row in rows[1:]} if rows else set()
    if str(student_id) not in ids:
        with open(STUDENTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, name])


def main(student_id: int = None, name: str = None) -> None:
    if student_id is None or name is None:
        try:
            student_id = int(input("Enter numeric ID (e.g. 1,2,3): ").strip())
            name = input("Enter name: ").strip()
        except ValueError:
            print("Invalid ID.")
            return

    add_student_to_csv(student_id, name)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Capturing faces. Press 'q' to quit early.")
    count = 0
    target_count = 40  # images per person

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
        )

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y : y + h, x : x + w]
            face_img = preprocess_face(face_img)

            filename = f"user.{student_id}.{count}.jpg"
            filepath = os.path.join(DATASET_DIR, filename)
            cv2.imwrite(filepath, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Samples: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Capture Faces - q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if count >= target_count:
            break

    print(f"Saved {count} images for ID {student_id} ({name}).")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

