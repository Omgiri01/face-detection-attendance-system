import os
import sqlite3


DB_PATH = os.path.join("data", "attendance.db")


def main() -> None:
    if not os.path.exists(DB_PATH):
        print("Database not found at", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("Students table:")
    for row in cur.execute("SELECT id, roll_no, name, branch FROM students ORDER BY id;"):
        print(row)

    print("\nAttendance table:")
    for row in cur.execute(
        "SELECT id, student_id, date, time, roll_no, name, branch FROM attendance ORDER BY date, time;"
    ):
        print(row)

    conn.close()


if __name__ == "__main__":
    main()

