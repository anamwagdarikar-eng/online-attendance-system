from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from src.database import save_student
    from src.face_utils import ensure_single_face
except ModuleNotFoundError:
    from database import save_student
    from face_utils import ensure_single_face


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a student and save their face embedding to PostgreSQL.")
    parser.add_argument("--student_id", required=True, help="Unique student ID")
    parser.add_argument("--full_name", required=True, help="Student full name")
    parser.add_argument("--email", default=None, help="Optional email")
    parser.add_argument("--department", default=None, help="Optional department")
    parser.add_argument("--program", default=None, help="Optional program")
    parser.add_argument("--year_level", default=None, help="Optional year level")
    parser.add_argument("--image", required=True, help="Path to a front-facing student photo")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    embedding = ensure_single_face(image_path)
    photo_path = str(Path(image_path).resolve())

    student_id = save_student(
        student_id=args.student_id,
        full_name=args.full_name,
        email=args.email,
        department=args.department,
        program=args.program,
        year_level=args.year_level,
        photo_path=photo_path,
        embedding=embedding.tolist(),
    )

    print(f"Student registered successfully with database ID: {student_id}")


if __name__ == "__main__":
    main()
