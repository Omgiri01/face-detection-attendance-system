import csv
import os
import sqlite3
from datetime import datetime

from db import DB_PATH, get_connection


OUTPUT_DIR = os.path.join("data", "attendance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_all() -> str:
    """Export all attendance records with full metadata to a timestamped CSV."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("attendance.db not found.")

    conn: sqlite3.Connection = get_connection()
    cur = conn.cursor()
    # Select from attendance table (which now contains metadata)
    cur.execute(
        """
        SELECT id, student_id, date, time, roll_no, name, branch
        FROM attendance
        ORDER BY date DESC, time DESC, id DESC;
        """
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No attendance records to export.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"attendance_full_export_{timestamp}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Record_ID", "Student_ID", "Date", "Time", "Roll_No", "Name", "Branch"])
        writer.writerows(rows)

    return out_path


if __name__ == "__main__":
    try:
        path = export_all()
        print(f"Attendance exported to {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Error exporting attendance: {exc}")

