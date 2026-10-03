# Online Attendance System

A Python + Streamlit attendance application that registers student photos into PostgreSQL, converts each face to a 128D embedding, and scans a classroom image or camera frame to mark attendance for multiple students at the same time.

## Features

- Student registration with photo upload
- Face embedding generation using `face_recognition`
- PostgreSQL storage using a `vector(128)` field via `pgvector`
- Group/class attendance scanning from a webcam or CCTV feed
- Attendance review and export-ready reporting

## Project structure

- `app.py` – Streamlit web app
- `src/config.py` – environment configuration
- `src/database.py` – PostgreSQL schema and queries
- `src/face_utils.py` – face encoding and matching logic
- `scripts/register_student.py` – command-line registration utility

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file using the provided Neon connection string:

   ```bash
   copy .env.example .env
   ```

3. Initialize the PostgreSQL schema:

   The app will create the needed tables when you click the database setup button in the Streamlit UI.

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Register students

Use the Streamlit dashboard or run the CLI:

```bash
python scripts/register_student.py \
  --student_id S001 \
  --full_name "Alice Johnson" \
  --email alice@example.com \
  --department "Computer Science" \
  --program "BSc Computer Science" \
  --year_level "2" \
  --image "sample/student1.jpg"
```

## Scan attendance

In the Streamlit app:

- Go to the Attendance Scanner tab
- Enter a webcam index like `0` or an RTSP/CCTV URL
- Click `Scan class now`
- The system will detect faces in the frame, match them against registered embeddings, and record attendance automatically

> This project is designed for classroom-level face recognition and works best with high-quality camera coverage and a clear view of the class.
