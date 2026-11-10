import os
import sqlite3


DB_PATH = os.path.join("data", "attendance.db")


def main() -> None:
    if not os.path.exists(DB_PATH):
        print("Database not found at", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance;")
    conn.commit()
    conn.close()
    print("All attendance records have been cleared.")


if __name__ == "__main__":
    main()

