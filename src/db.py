import csv
import os
import sqlite3
from datetime import datetime
from typing import Dict, Tuple


DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
STUDENTS_CSV = os.path.join(DATA_DIR, "students.csv")


def get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            roll_no TEXT,
            name TEXT NOT NULL,
            branch TEXT
        );
        """
    )

    # If the students table existed before without roll_no, ensure the column exists
    # and is populated (defaulting roll_no to the id value).
    cur.execute("PRAGMA table_info(students);")
    cols = [row[1] for row in cur.fetchall()]
    if "roll_no" not in cols:
        cur.execute("ALTER TABLE students ADD COLUMN roll_no TEXT;")
        cur.execute("UPDATE students SET roll_no = CAST(id AS TEXT) WHERE roll_no IS NULL;")
    if "branch" not in cols:
        cur.execute("ALTER TABLE students ADD COLUMN branch TEXT;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            roll_no TEXT,
            name TEXT,
            branch TEXT,
            UNIQUE(student_id, date),
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        """
    )

    # Table for deep-model embeddings (e.g., DeepFace / Facenet512).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        """
    )

    # Ensure roll_no and name columns exist in attendance table for easier inspection.
    cur.execute("PRAGMA table_info(attendance);")
    att_cols = [row[1] for row in cur.fetchall()]
    if "roll_no" not in att_cols:
        cur.execute("ALTER TABLE attendance ADD COLUMN roll_no TEXT;")
    if "name" not in att_cols:
        cur.execute("ALTER TABLE attendance ADD COLUMN name TEXT;")
    if "branch" not in att_cols:
        cur.execute("ALTER TABLE attendance ADD COLUMN branch TEXT;")

    # Backfill roll_no and name for any existing attendance rows.
    cur.execute(
        """
        UPDATE attendance
        SET roll_no = (
                SELECT roll_no FROM students WHERE students.id = attendance.student_id
            ),
            name = (
                SELECT name FROM students WHERE students.id = attendance.student_id
            ),
            branch = (
                SELECT branch FROM students WHERE students.id = attendance.student_id
            )
        WHERE roll_no IS NULL OR name IS NULL OR branch IS NULL;
        """
    )

    conn.commit()
    conn.close()


def sync_students_from_csv() -> None:
    """Load students from students.csv into the SQLite students table."""
    if not os.path.exists(STUDENTS_CSV):
        return

    conn = get_connection()
    cur = conn.cursor()

    with open(STUDENTS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize headers to handle accidental spaces or case changes
        headers = {k.lower().strip(): k for k in reader.fieldnames}
        
        # Mapping for common Google Form / manual headers
        id_key = headers.get('id') or headers.get('roll number') or headers.get('roll_no')
        name_key = headers.get('name') or headers.get('full name')
        roll_key = headers.get('roll_no') or headers.get('roll number')
        branch_key = headers.get('branch') or headers.get('branch ') # Handles trailing space

        for row in reader:
            try:
                if not name_key or not id_key:
                    continue
                
                # Get student ID from id_key or roll_key
                try:
                    student_id = int(row[id_key])
                except (ValueError, TypeError):
                    continue

                name = row[name_key]
                roll_no = row.get(roll_key, str(student_id))
                branch = row.get(branch_key, "")

                cur.execute(
                    """
                    INSERT INTO students (id, roll_no, name, branch)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        roll_no = excluded.roll_no,
                        name    = excluded.name,
                        branch  = excluded.branch;
                    """,
                    (student_id, roll_no, name, branch),
                )
            except (KeyError, ValueError, sqlite3.Error) as e:
                print(f"Skipping row due to error: {e}")
                continue

    conn.commit()
    conn.close()


def load_students_from_db() -> Dict[int, str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM students;")
    rows = cur.fetchall()
    conn.close()
    return {int(row[0]): str(row[1]) for row in rows}


def load_student_details() -> Dict[int, Dict[str, str]]:
    """Returns a full dictionary of student details for recognition labels."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, roll_no, branch FROM students;")
    rows = cur.fetchall()
    conn.close()
    return {
        int(row[0]): {
            "name": str(row[1]),
            "roll_no": str(row[2] if row[2] else row[0]),
            "branch": str(row[3] if row[3] else "N/A")
        } 
        for row in rows
    }


def mark_attendance_if_needed(student_id: int) -> Tuple[bool, datetime]:
    """
    Insert attendance for student_id for today if not already present.
    Returns (created, timestamp_used).
    """
    now = datetime.now()
    today_str = now.date().isoformat()
    time_str = now.strftime("%H:%M:%S")

    conn = get_connection()
    cur = conn.cursor()

    # Skip insert if student already has attendance for today.
    cur.execute(
        """
        SELECT 1 FROM attendance
        WHERE student_id = ? AND date = ?;
        """,
        (student_id, today_str),
    )
    exists = cur.fetchone() is not None

    created = False
    if not exists:
        cur.execute(
            "SELECT roll_no, name, branch FROM students WHERE id = ?;",
            (student_id,),
        )
        row = cur.fetchone()
        roll_no_val = row[0] if row else str(student_id)
        name_val = row[1] if row else ""
        branch_val = row[2] if row else ""

        cur.execute(
            """
            INSERT INTO attendance (student_id, date, time, roll_no, name, branch)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (student_id, today_str, time_str, roll_no_val, name_val, branch_val),
        )
        conn.commit()
        created = True

    conn.close()
    return created, now


