from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

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


def normalize_rtsp_url(candidate: str) -> str:
    raw = candidate.strip()
    if not raw.lower().startswith("rtsp://"):
        return raw

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw

    if not parsed.hostname:
        return raw

    username = parsed.username or ""
    password = parsed.password or ""
    if not username and not password:
        return raw

    userinfo = quote(username, safe="")
    if password:
        userinfo = f"{userinfo}:{quote(password, safe='')}"

    host_port = parsed.hostname
    if parsed.port:
        host_port = f"{host_port}:{parsed.port}"

    rebuilt = urlunsplit((parsed.scheme, f"{userinfo}@{host_port}", parsed.path, parsed.query, parsed.fragment))
    if not rebuilt:
        return raw

    query = parsed.query
    if query:
        parts = [part for part in query.split("&") if part]
        if not any(part.lower() == "tcp" or part.lower().startswith("tcp=") for part in parts):
            rebuilt = f"{rebuilt}&tcp"
    else:
        rebuilt = f"{rebuilt}?tcp"

    return rebuilt


def build_rtsp_url_candidates(candidate: str) -> list[str]:
    raw = candidate.strip()
    if not raw.lower().startswith("rtsp://"):
        return [raw]

    normalized = normalize_rtsp_url(raw)
    candidates = [raw, normalized]
    seen = set()
    ordered: list[str] = []
    for url in candidates:
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            ordered.append(url)
        if "tcp" not in url.lower():
            if "?" in url:
                tcp_url = f"{url}&tcp"
            else:
                tcp_url = f"{url}?tcp"
            if tcp_url not in seen:
                seen.add(tcp_url)
                ordered.append(tcp_url)
    return ordered


def try_open_rtsp_stream(url: str):
    candidates = build_rtsp_url_candidates(url)
    if not candidates or not candidates[0]:
        return False, candidates[0] if candidates else "", "RTSP URL is empty."

    for candidate in candidates:
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(candidate, backend)
            except Exception:
                continue
            if cap is None or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return True, candidate, "Stream opened successfully."

    return False, candidates[0], "Failed to open stream. Check the IP, username, password, stream path, port 554, and network access."


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
    st.info("Use the six-step capture workflow below. Capture 5 to 6 clear photos from different angles for better recognition accuracy.")
    student_id = st.text_input("Student ID")
    full_name = st.text_input("Full name")
    email = st.text_input("Email")
    department = st.text_input("Department")
    program = st.text_input("Program")
    year_level = st.text_input("Year level")
    class_name = st.text_input("Class name")

    uploaded_photos = st.file_uploader(
        "Optional: upload extra photos from different angles",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if "angle_captures" not in st.session_state:
        st.session_state.angle_captures = {}

    for angle in range(1, 7):
        key = f"angle_{angle}"
        photo = st.camera_input(f"Capture angle {angle}", key=key)
        if photo is not None:
            st.session_state.angle_captures[angle] = photo
        if angle in st.session_state.angle_captures:
            st.image(st.session_state.angle_captures[angle], caption=f"Angle {angle} captured", width=220)

    selected_photos = []
    if uploaded_photos:
        selected_photos.extend(uploaded_photos)
    for angle in range(1, 7):
        if angle in st.session_state.angle_captures:
            selected_photos.append(st.session_state.angle_captures[angle])

    if selected_photos:
        st.caption(f"{len(selected_photos)} photo(s) selected for registration")
        cols = st.columns(min(6, len(selected_photos)))
        for idx, photo in enumerate(selected_photos[:6]):
            with cols[idx % len(cols)]:
                st.image(photo, caption=f"Selected angle {idx + 1}")

    if st.button("Register student"):
        if not student_id or not full_name:
            st.error("Please provide a student ID and full name.")
        elif len(selected_photos) < 5:
            st.error("Please capture or upload at least 5 photos from different angles before registering.")
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
                        raise ValueError("No valid face descriptors were generated from the selected photos.")

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
            cv2.CAP_ANY,
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
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
                ret = False
                try:
                    ret, _ = cap.read()
                    if ret:
                        return cap
                except Exception:
                    continue
                finally:
                    if not ret:
                        try:
                            cap.release()
                        except Exception:
                            pass

        if raw and not raw.isdigit():
            for raw_url in build_rtsp_url_candidates(raw):
                for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
                    try:
                        cap = cv2.VideoCapture(raw_url, backend)
                    except Exception:
                        continue
                    if cap is None or not cap.isOpened():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        continue
                    ret = False
                    try:
                        ret, _ = cap.read()
                        if ret:
                            return cap
                    except Exception:
                        pass
                    finally:
                        if not ret:
                            try:
                                cap.release()
                            except Exception:
                                pass
                if raw_url.lower().startswith("rtsp://"):
                    try:
                        cap = cv2.VideoCapture(raw_url)
                        if cap is not None and cap.isOpened():
                            ret = False
                            try:
                                ret, _ = cap.read()
                                if ret:
                                    return cap
                            except Exception:
                                pass
                            finally:
                                if not ret:
                                    try:
                                        cap.release()
                                    except Exception:
                                        pass
                    except Exception:
                        pass

        return None

    st.info("OpenCV can only access a device when the OS exposes it to the Python process. If your laptop webcam or phone camera is not visible to OpenCV, use the browser camera below; it works from the browser and is the most reliable fallback.")

    browser_snapshot = st.camera_input("Use browser camera (recommended)")

    def prepare_frame(frame: np.ndarray, max_dimension: int = 1200) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(1.0, max_dimension / max(height, width))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return frame

    def render_live_preview(cap: cv2.VideoCapture, title: str = "Live preview", duration_seconds: float = 8.0) -> bool:
        if cap is None or not cap.isOpened():
            return False

        placeholder = st.empty()
        deadline = time.monotonic() + duration_seconds
        frames_shown = 0
        while cap.isOpened() and time.monotonic() < deadline:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            frame = prepare_frame(frame)
            placeholder.image(frame, channels="BGR", caption=title)
            frames_shown += 1
            if frames_shown >= 20:
                break

        try:
            cap.release()
        except Exception:
            pass
        return True

    if browser_snapshot is not None:
        current_browser_hash = hash(browser_snapshot.getvalue())
        last_browser_hash = st.session_state.get("last_browser_hash")
        if current_browser_hash != last_browser_hash:
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
                    st.session_state["last_browser_hash"] = current_browser_hash
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
            if ret and frame is not None:
                st.success("Camera is working and a frame was captured successfully.")
                st.caption("Live camera preview")
                render_live_preview(cap, title="Live camera preview", duration_seconds=8.0)
            else:
                try:
                    cap.release()
                except Exception:
                    pass
                st.warning("Camera is present but no frame was readable. Check permission, driver, or the stream URL.")

    if st.button("Check CCTV stream"):
        url_to_test = source.strip()
        if not url_to_test:
            st.warning("Please enter a CCTV RTSP URL first.")
        else:
            ok, exact_url, message = try_open_rtsp_stream(url_to_test)
            st.code(exact_url)
            if ok:
                st.success(f"CCTV stream opened successfully: {message}")
                preview_cap = open_camera_candidate(url_to_test)
                if preview_cap is not None:
                    st.caption("Live RTSP preview")
                    render_live_preview(preview_cap, title="Live RTSP preview", duration_seconds=8.0)
            else:
                st.error(f"CCTV stream failed: {message}")

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
