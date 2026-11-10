import os
import subprocess
import sys


# src directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# project root is the parent of src
PROJECT_ROOT = os.path.dirname(SRC_DIR)
PYTHON = sys.executable


def run(script_name: str) -> None:
    """Run another Python script from the project root and wait for it to finish."""
    cmd = [PYTHON, os.path.join("src", script_name)]
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def main() -> None:
    while True:
        print("\n==== FACE ATTENDANCE SYSTEM MENU ====")
        print("1. Enrol new student (capture faces)")
        print("2. Augment dataset (generate extra images)")
        print("3. Train recognition model (LBPH)")
        print("4. Take attendance (live recognition)")
        print("5. Show database tables")
        print("6. Export today's attendance CSV")
        print("7. Clear all attendance records (testing only)")
        print("8. Full model analysis (train/test split, report)")
        print("0. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            run("capture_faces.py")
        elif choice == "2":
            run("augment_dataset.py")
        elif choice == "3":
            run("train_model.py")
        elif choice == "4":
            run("recognize_and_mark_attendance.py")
        elif choice == "5":
            run("show_db.py")
        elif choice == "6":
            run("export_attendance.py")
        elif choice == "8":
            run("full_model_analysis.py")
        elif choice == "7":
            confirm = input(
                "This will delete ALL attendance records. Continue (y/n)? "
            ).strip().lower()
            if confirm == "y":
                run("clear_attendance.py")
        elif choice == "0":
            print("Exiting menu.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

