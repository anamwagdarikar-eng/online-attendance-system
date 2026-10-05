from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover
    letter = None
    canvas = None

PROJECT_ROOT = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT.parent):
    if candidate and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

src_dir = PROJECT_ROOT / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.database import fetch_recent_attendance, init_db, insert_attendance_record
from src.face_utils import annotated_frame, detect_and_match_faces, ensure_single_face, save_uploaded_file

st.set_page_config(page_title="Online Attendance System", layout="wide")


@st.cache_data
def get_student_summary() -> pd.DataFrame:
    records = fetch_recent_attendance(limit=100)
    return pd.DataFrame(records)


def enumerate_local_cameras(max_index: int = 10) -> list[int]:
    backend_list = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    detected: list[int] = []
    seen: set[tuple[int, int]] = set()

    for index in range(max_index):
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
            try:
                ok, _ = cap.read()
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
            if ok:
                detected.append(index)
                break
    return detected


def render_attendance_pdf(df: pd.DataFrame) -> bytes:
    if canvas is None:
        raise RuntimeError("reportlab is not installed.")

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    y = 760
    pdf.setTitle("Attendance Record")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Attendance Report")
    y -= 24
    pdf.setFont("Helvetica", 10)

    if df.empty:
        pdf.drawString(50, y, "No attendance records yet.")
    else:
        columns = ["student_id", "student_name", "class_name", "attendance_date", "attendance_time", "confidence", "source"]
        visible_cols = [col for col in columns if col in df.columns]
        for _, row in df.head(25).iterrows():
            if y < 80:
                pdf.showPage()
                y = 760
            values = [str(row.get(col, "")) for col in visible_cols]
            pdf.drawString(50, y, " | ".join(values))
            y -= 18

    pdf.save()
    return buffer.getvalue()


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
    st.info("Upload 5 to 6 photos of the same student from different angles, or capture them with the browser camera. This improves recognition accuracy during class scans.")
    student_id = st.text_input("Student ID")
    full_name = st.text_input("Full name")
    email = st.text_input("Email")
    department = st.text_input("Department")
    program = st.text_input("Program")
    year_level = st.text_input("Year level")
    class_name = st.text_input("Class name")

    uploaded_photos = st.file_uploader(
        "Upload 5–6 student photos from different angles",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    captured_photo = st.camera_input("Capture one angle with the browser camera")

    selected_photos = []
    if uploaded_photos:
        selected_photos.extend(uploaded_photos)
    if captured_photo is not None:
        selected_photos.append(captured_photo)

    if selected_photos:
        st.caption(f"{len(selected_photos)} photo(s) selected for registration")
        cols = st.columns(min(6, len(selected_photos)))
        for idx, photo in enumerate(selected_photos[:6]):
            with cols[idx % len(cols)]:
                st.image(photo, caption=f"Angle {idx + 1}")

    if st.button("Register student"):
        if not student_id or not full_name:
            st.error("Please provide a student ID and full name.")
        elif len(selected_photos) < 5:
            st.error("Please upload or capture at least 5 photos from different angles before registering.")
        else:
            try:
                descriptors = []
                temp_paths = []
                try:
                    for photo in selected_photos[:6]:
                        photo.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                            temp_file.write(photo.getvalue())
                            temp_path = temp_file.name
                        temp_paths.append(temp_path)
                        descriptors.append(ensure_single_face(temp_path))

                    if not descriptors:
                        raise ValueError("No valid face descriptors were generated from the uploaded photos.")

                    embedding_array = np.vstack(descriptors)
                    average_embedding = np.mean(embedding_array, axis=0)
                    from src.database import save_student

                    saved_id = save_student(
                        student_id=student_id,
                        full_name=full_name,
                        email=email or None,
                        department=department or None,
                        program=program or None,
                        year_level=year_level or None,
                        photo_path=temp_paths[0],
                        embedding=average_embedding.astype(float).tolist(),
                    )
                    st.success(f"Student {full_name} registered successfully with ID {saved_id} using {len(descriptors)} angle photos.")
                finally:
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            except Exception as exc:
                st.error(f"Registration failed: {exc}")

with attendance_tab:
    st.subheader("2) Scan the entire class")
    detected_local_cameras = enumerate_local_cameras()
    if detected_local_cameras:
        st.caption(f"Detected OpenCV local cameras: {detected_local_cameras}")
        default_source = str(detected_local_cameras[0])
    else:
        default_source = "0"

    source = st.text_input("Camera index or CCTV RTSP URL", value=default_source)
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

    st.info("OpenCV can only access a device when the OS exposes it to the Python process. If your laptop webcam or phone camera is not visible to OpenCV, use the browser camera below; it works from the browser and is the most reliable fallback.")

    browser_snapshot = st.camera_input("Use browser camera (recommended)")

    def prepare_frame(frame: np.ndarray, max_dimension: int = 1200) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(1.0, max_dimension / max(height, width))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return frame

    if browser_snapshot is not None:
        try:
            file_bytes = browser_snapshot.getvalue()
            image_array = np.frombuffer(file_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                st.error("The browser camera image could not be read.")
            else:
                frame = prepare_frame(frame)
                recognized = detect_and_match_faces(frame, tolerance=tolerance)
                st.session_state["last_recognized"] = recognized
                if not recognized:
                    st.warning("No registered faces were found in the browser camera image.")
                processed = annotated_frame(frame, recognized)
                st.image(processed, channels="BGR", caption="Browser camera snapshot with detected students")

                if recognized:
                    rows = [{
                        "Student ID": item["student_id"],
                        "Name": item["full_name"],
                        "Confidence": round(item["confidence"], 3),
                    } for item in recognized]
                    st.dataframe(rows)
                    if st.button("Submit attendance", key="submit_browser"):
                        for match in recognized:
                            insert_attendance_record(
                                student_id=match["student_id"],
                                student_name=match["full_name"],
                                class_name=class_name,
                                confidence=match["confidence"],
                                source="browser_camera",
                            )
                        st.success(f"Submitted attendance for {len(recognized)} detected student(s).")
        except Exception as exc:
            st.error(f"Browser camera processing failed: {exc}")

    if st.button("Test camera"):
        cap = open_camera_candidate(source.strip())
        if cap is None:
            st.warning(
                "No local camera was detected by OpenCV. This usually means the webcam is blocked or unavailable to the machine, not that the hardware is missing. "
                "Use the browser camera above for a reliable fallback."
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
                    st.session_state["last_recognized"] = recognized
                    if not recognized:
                        st.warning("No registered faces were found in the uploaded class image.")
                    processed = annotated_frame(frame, recognized)
                    st.image(processed, channels="BGR", caption="Uploaded classroom image with detected students")

                    if recognized:
                        rows = [{
                            "Student ID": item["student_id"],
                            "Name": item["full_name"],
                            "Confidence": round(item["confidence"], 3),
                        } for item in recognized]
                        st.dataframe(rows)
                        if st.button("Submit attendance", key="submit_upload"):
                            for match in recognized:
                                insert_attendance_record(
                                    student_id=match["student_id"],
                                    student_name=match["full_name"],
                                    class_name=class_name,
                                    confidence=match["confidence"],
                                    source="image_upload",
                                )
                            st.success(f"Submitted attendance for {len(recognized)} detected student(s).")
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
                    frame = prepare_frame(frame)
                    recognized = detect_and_match_faces(frame, tolerance=tolerance)
                    st.session_state["last_recognized"] = recognized
                    if not recognized:
                        st.warning("No registered faces were found in the captured frame.")
                    processed = annotated_frame(frame, recognized)
                    st.image(processed, channels="BGR", caption="Captured classroom frame with detected students")

                    if recognized:
                        rows = [{
                            "Student ID": item["student_id"],
                            "Name": item["full_name"],
                            "Confidence": round(item["confidence"], 3),
                        } for item in recognized]
                        st.dataframe(rows)
                        if st.button("Submit attendance", key="submit_camera"):
                            for match in recognized:
                                insert_attendance_record(
                                    student_id=match["student_id"],
                                    student_name=match["full_name"],
                                    class_name=class_name,
                                    confidence=match["confidence"],
                                    source="camera",
                                )
                            st.success(f"Submitted attendance for {len(recognized)} detected student(s).")

with records_tab:
    st.subheader("3) Attendance log")
    df = get_student_summary()
    if df.empty:
        st.info("No attendance records yet.")
    else:
        st.dataframe(df, width="stretch")
        excel_bytes = None
        try:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_bytes = excel_buffer.getvalue()
        except Exception as exc:
            st.warning(f"Excel export is unavailable: {exc}")

        if excel_bytes is not None:
            st.download_button(
                "Download attendance as Excel",
                data=excel_bytes,
                file_name="attendance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        try:
            pdf_bytes = render_attendance_pdf(df)
            st.download_button(
                "Download attendance as PDF",
                data=pdf_bytes,
                file_name="attendance.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.warning(f"PDF export is unavailable: {exc}")
