from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent):
    if candidate and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from src.database import fetch_recent_attendance, init_db, insert_attendance_record
    from src.face_utils import annotated_frame, detect_and_match_faces, ensure_single_face, save_uploaded_file
except ModuleNotFoundError:
    if (PROJECT_ROOT / "database.py").exists() and (PROJECT_ROOT / "face_utils.py").exists():
        from database import fetch_recent_attendance, init_db, insert_attendance_record
        from face_utils import annotated_frame, detect_and_match_faces, ensure_single_face, save_uploaded_file
    else:
        src_dir = PROJECT_ROOT / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from database import fetch_recent_attendance, init_db, insert_attendance_record
        from face_utils import annotated_frame, detect_and_match_faces, ensure_single_face, save_uploaded_file

st.set_page_config(page_title="Online Attendance System", layout="wide")


@st.cache_data
def get_student_summary() -> pd.DataFrame:
    records = fetch_recent_attendance(limit=100)
    return pd.DataFrame(records)


st.title("Online Attendance System")
st.caption("Face registration + classroom attendance scanning using Python, Streamlit, and PostgreSQL")

with st.sidebar:
    st.header("System Setup")
    if st.button("Initialize PostgreSQL schema"):
        init_db()
        st.success("Database tables are ready.")

    st.markdown("Database URL is loaded from the environment or defaults to your Neon connection string.")


registration_tab, attendance_tab, records_tab = st.tabs(["Student Registration", "Attendance Scanner", "Attendance Records"])

with registration_tab:
    st.subheader("1) Register a student face")
    student_id = st.text_input("Student ID")
    full_name = st.text_input("Full name")
    email = st.text_input("Email")
    department = st.text_input("Department")
    program = st.text_input("Program")
    year_level = st.text_input("Year level")
    class_name = st.text_input("Class name")
    uploaded_photo = st.file_uploader("Upload a front-facing student photo", type=["jpg", "jpeg", "png"])

    if uploaded_photo is not None:
        temp_path = save_uploaded_file(uploaded_photo)
        try:
            encoding = ensure_single_face(temp_path)
            st.image(uploaded_photo, caption="Uploaded student photo")
            st.success(f"Face encoded successfully: {len(encoding)} features detected")
        except Exception as exc:
            st.error(str(exc))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if st.button("Register student"):
        if not student_id or not full_name or uploaded_photo is None:
            st.error("Please provide a student ID, full name, and a photo.")
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(uploaded_photo.read())
                    temp_path = temp_file.name

                embedding = ensure_single_face(temp_path)
                try:
                    from src.database import save_student
                except ModuleNotFoundError:
                    from database import save_student

                saved_id = save_student(
                    student_id=student_id,
                    full_name=full_name,
                    email=email or None,
                    department=department or None,
                    program=program or None,
                    year_level=year_level or None,
                    photo_path=temp_path,
                    embedding=embedding.tolist(),
                )

                st.success(f"Student {full_name} registered successfully with ID {saved_id}.")
            except Exception as exc:
                st.error(f"Registration failed: {exc}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

with attendance_tab:
    st.subheader("2) Scan the entire class")
    source = st.text_input("Camera index or CCTV RTSP URL", value="0")
    class_name = st.text_input("Attendance class name", value="CS-101")
    tolerance = st.slider("Face match tolerance", min_value=0.25, max_value=0.6, value=0.45, step=0.01)
    uploaded_class_image = st.file_uploader("Or upload a classroom image", type=["jpg", "jpeg", "png"])

    def open_camera_candidate(candidate: str):
        try:
            idx = int(candidate)
        except ValueError:
            idx = None

        attempts = []
        if idx is not None:
            attempts = [str(idx)] + [str(i) for i in range(0, 5) if i != idx]
        else:
            attempts = [candidate]

        for attempt in attempts:
            cap = cv2.VideoCapture(attempt)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    if st.button("Scan class now"):
        if uploaded_class_image is not None:
            temp_path = save_uploaded_file(uploaded_class_image)
            try:
                frame = cv2.imread(temp_path)
                if frame is None:
                    st.error("Could not read the uploaded class image.")
                else:
                    recognized = detect_and_match_faces(frame, tolerance=tolerance)
                    if not recognized:
                        st.warning("No registered faces were found in the uploaded class image.")
                    else:
                        for match in recognized:
                            insert_attendance_record(
                                student_id=match["student_id"],
                                student_name=match["full_name"],
                                class_name=class_name,
                                confidence=match["confidence"],
                                source="image_upload",
                            )
                            st.success(f"Marked attendance for {match['full_name']} ({match['student_id']})")

                    processed = annotated_frame(frame, recognized)
                    st.image(processed, channels="BGR", caption="Uploaded classroom image with detected students")

                    if recognized:
                        rows = [{
                            "Student ID": item["student_id"],
                            "Name": item["full_name"],
                            "Confidence": round(item["confidence"], 3),
                        } for item in recognized]
                        st.dataframe(rows)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            cap = open_camera_candidate(source.strip())
            if cap is None:
                st.error(
                    "No camera was found for this source. This usually means the machine has no webcam connected, or the RTSP stream is invalid. "
                    "Use a local camera index like 0, a valid RTSP URL, or upload a classroom image instead."
                )
            else:
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    st.error("The camera opened but no frame could be read. Check the device, permission, or RTSP stream.")
                else:
                    recognized = detect_and_match_faces(frame, tolerance=tolerance)
                    if not recognized:
                        st.warning("No registered faces were found in the captured frame.")
                    else:
                        for match in recognized:
                            insert_attendance_record(
                                student_id=match["student_id"],
                                student_name=match["full_name"],
                                class_name=class_name,
                                confidence=match["confidence"],
                                source="camera",
                            )
                            st.success(f"Marked attendance for {match['full_name']} ({match['student_id']})")

                    processed = annotated_frame(frame, recognized)
                    st.image(processed, channels="BGR", caption="Captured classroom frame with detected students")

                    if recognized:
                        rows = [{
                            "Student ID": item["student_id"],
                            "Name": item["full_name"],
                            "Confidence": round(item["confidence"], 3),
                        } for item in recognized]
                        st.dataframe(rows)

with records_tab:
    st.subheader("3) Attendance log")
    df = get_student_summary()
    if df.empty:
        st.info("No attendance records yet.")
    else:
        st.dataframe(df, use_container_width=True)
