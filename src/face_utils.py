from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from src.database import fetch_known_students


def _find_haarcascade_xml() -> Optional[str]:
    names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt2.xml",
    ]

    roots = []
    for base in [sys.prefix, sys.base_prefix]:
        if base:
            roots.append(Path(base))

    for root in roots:
        for candidate in [
            root / "Lib" / "site-packages" / "cv2" / "data" / "haarcascades",
            root / "Lib" / "site-packages" / "opencv" / "data" / "haarcascades",
            root / "lib" / "python" / "site-packages" / "cv2" / "data" / "haarcascades",
            root / "lib" / "python3" / "site-packages" / "cv2" / "data" / "haarcascades",
            root / "lib" / "site-packages" / "cv2" / "data" / "haarcascades",
            root / "lib" / "site-packages" / "opencv" / "data" / "haarcascades",
        ]:
            for name in names:
                path = candidate / name
                if path.exists():
                    return str(path)

    for root in [Path(__file__).resolve().parents[1], Path(sys.executable).resolve().parent.parent]:
        if root.exists():
            for pattern in [
                "**/haarcascade_frontalface_default.xml",
                "**/haarcascade_frontalface_alt.xml",
                "**/haarcascade_frontalface_alt2.xml",
            ]:
                matches = sorted(root.glob(pattern))
                if matches:
                    return str(matches[0])

    return None


def get_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = _find_haarcascade_xml()
    if cascade_path is None:
        raise RuntimeError(
            "OpenCV face cascade file was not found. Reinstall opencv-python-headless or opencv-python so the Haar cascade XML is available."
        )

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"OpenCV failed to load the face cascade file: {cascade_path}")
    return cascade


def _descriptor_from_face(face_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(float)
    return hist


def ensure_single_face(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    cascade = get_face_cascade()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly one face in {image_path}, but found {len(faces)}. "
            "Please use a single face photo for registration."
        )

    x, y, w, h = faces[0]
    face_crop = image[y : y + h, x : x + w]
    return _descriptor_from_face(face_crop)


def save_uploaded_file(uploaded_file: Any, folder: str = "uploads") -> str:
    os.makedirs(folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=folder) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    return temp_path


def find_best_match(face_embedding: np.ndarray, tolerance: float = 0.72) -> Optional[dict[str, Any]]:
    known_students = fetch_known_students()
    if not known_students:
        return None

    best_match = None
    best_similarity = -1.0

    for student in known_students:
        known_embedding = np.asarray(student["embedding"], dtype=float)
        dot = float(np.dot(face_embedding, known_embedding))
        norm = float(np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))
        similarity = dot / (norm + 1e-9)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = student

    if best_match and best_similarity >= tolerance:
        return {
            "student_id": best_match["student_id"],
            "full_name": best_match["full_name"],
            "confidence": float(best_similarity),
        }

    return None


def detect_and_match_faces(frame: np.ndarray, tolerance: float = 0.72) -> List[dict[str, Any]]:
    cascade = get_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return []

    results: List[dict[str, Any]] = []
    for (x, y, w, h) in faces:
        face_crop = frame[y : y + h, x : x + w]
        descriptor = _descriptor_from_face(face_crop)
        match = find_best_match(descriptor, tolerance=tolerance)
        if match:
            results.append(
                {
                    "student_id": match["student_id"],
                    "full_name": match["full_name"],
                    "confidence": match["confidence"],
                    "bbox": (y, x + w, y + h, x),
                }
            )

    return results


def annotated_frame(frame: np.ndarray, recognized: List[dict[str, Any]]) -> np.ndarray:
    image = frame.copy()
    for match in recognized:
        top, right, bottom, left = match["bbox"]
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{match['full_name']} ({match['confidence']:.2f})"
        cv2.putText(
            image,
            label,
            (left, max(top - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return image
