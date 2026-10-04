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
        uploaded_photo.seek(0)
        temp_path = save_uploaded_file(uploaded_photo)
        try:
            encoding = ensure_single_face(temp_path)
            uploaded_photo.seek(0)
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
                uploaded_photo.seek(0)
                file_bytes = uploaded_photo.getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(file_bytes)
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
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)

with attendance_tab:
    st.subheader("2) Scan the entire class")
    source = st.text_input("Camera index or CCTV RTSP URL", value="0")
    class_name = st.text_input("Attendance class name", value="CS-101")
    tolerance = st.slider("Face match tolerance", min_value=0.25, max_value=0.6, value=0.45, step=0.01)
    uploaded_class_image = st.file_uploader("Or upload a classroom image", type=["jpg", "jpeg", "png"])

    def open_camera_candidate(candidate: str):
        raw = candidate.strip()
        numeric_candidates = []

        if raw:
            try:
                numeric_candidates.append(int(raw))
            except ValueError:
                pass

        numeric_candidates.extend(range(0, 10))

        backend_list = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY,
        ]

        seen = set()
        for index in numeric_candidates:
            for backend in backend_list:
                key = (index, backend)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    cap = cv2.VideoCapture(index, backend)
                except Exception:
                    continue
                if cap is None or not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    continue

                ret, _ = cap.read()
                if ret:
                    return cap

                try:
                    cap.release()
                except Exception:
                    pass

        if raw and not raw.isdigit():
            for backend in backend_list:
                try:
                    cap = cv2.VideoCapture(raw, backend)
                except Exception:
                    continue
                if cap is not None and cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        return cap
                    cap.release()

        return None

    st.info("If the local OpenCV webcam is blocked, use the browser camera below. This bypasses Windows webcam permission and driver issues because the browser captures the camera directly.")

    browser_snapshot = st.camera_input("Use browser camera (recommended)")

    if browser_snapshot is not None:
        try:
            file_bytes = browser_snapshot.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(file_bytes)
                browser_path = temp_file.name

            frame = cv2.imread(browser_path)
            if frame is None:
                st.error("The browser camera image could not be read.")
            else:
                recognized = detect_and_match_faces(frame, tolerance=tolerance)
                if not recognized:
                    st.warning("No registered faces were found in the browser camera image.")
                else:
                    for match in recognized:
                        insert_attendance_record(
                            student_id=match["student_id"],
                            student_name=match["full_name"],
                            class_name=class_name,
                            confidence=match["confidence"],
                            source="browser_camera",
                        )
                        st.success(f"Marked attendance for {match['full_name']} ({match['student_id']})")

                processed = annotated_frame(frame, recognized)
                st.image(processed, channels="BGR", caption="Browser camera snapshot with detected students")

                if recognized:
                    rows = [{
                        "Student ID": item["student_id"],
                        "Name": item["full_name"],
                        "Confidence": round(item["confidence"], 3),
                    } for item in recognized]
                    st.dataframe(rows)
        except Exception as exc:
            st.error(f"Browser camera processing failed: {exc}")
        finally:
            if 'browser_path' in locals() and os.path.exists(browser_path):
                os.remove(browser_path)

    if st.button("Test camera"):
        cap = open_camera_candidate(source.strip())
        if cap is None:
            st.warning(
                "No camera was detected by OpenCV. This usually means the webcam is blocked or unavailable to the local machine. "
                "Use the browser camera above, which works even when OpenCV cannot access the device."
            )
        else:
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                st.success("Camera is working and a frame was captured successfully.")
                st.image(frame, channels="BGR", caption="Live camera preview")
            else:
                st.warning("Camera is present but no frame was readable. Check permission, driver, or the stream URL.")

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
