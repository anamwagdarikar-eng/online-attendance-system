from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from src.config import DATABASE_URL


@contextmanager
def get_connection() -> Iterator[psycopg.Connection[Any]]:
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def ensure_vector_extension() -> None:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    finally:
        conn.close()


def init_db() -> None:
    ensure_vector_extension()

    sql = """
    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        student_id VARCHAR(50) UNIQUE NOT NULL,
        full_name VARCHAR(255) NOT NULL,
        email VARCHAR(255),
        department VARCHAR(120),
        program VARCHAR(120),
        year_level VARCHAR(30),
        photo_path TEXT,
        face_embedding vector(128),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS attendance_records (
        id SERIAL PRIMARY KEY,
        student_id VARCHAR(50) NOT NULL,
        student_name VARCHAR(255) NOT NULL,
        class_name VARCHAR(120) NOT NULL,
        attendance_date DATE NOT NULL,
        attendance_time TIME NOT NULL,
        confidence FLOAT,
        source VARCHAR(80) DEFAULT 'camera',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def save_student(
    student_id: str,
    full_name: str,
    email: Optional[str],
    department: Optional[str],
    program: Optional[str],
    year_level: Optional[str],
    photo_path: str,
    embedding: List[float],
) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO students (
                    student_id, full_name, email, department, program, year_level, photo_path, face_embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (student_id)
                DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    email = EXCLUDED.email,
                    department = EXCLUDED.department,
                    program = EXCLUDED.program,
                    year_level = EXCLUDED.year_level,
                    photo_path = EXCLUDED.photo_path,
                    face_embedding = EXCLUDED.face_embedding
                RETURNING id
                """,
                (student_id, full_name, email, department, program, year_level, photo_path, embedding),
            )
            row = cur.fetchone()
        conn.commit()
        return int(row[0])


def _normalize_embedding(embedding: Any) -> List[float]:
    if embedding is None:
        return []

    if hasattr(embedding, "to_list"):
        values = embedding.to_list()
    elif hasattr(embedding, "tolist"):
        values = embedding.tolist()
    else:
        values = list(embedding)

    return [float(value) for value in values]


def fetch_known_students() -> List[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, student_id, full_name, face_embedding FROM students WHERE face_embedding IS NOT NULL"
            )
            rows = cur.fetchall()

    students: List[dict[str, Any]] = []
    for row in rows:
        student_id, full_name, embedding = row[1], row[2], row[3]
        students.append({
            "id": row[0],
            "student_id": student_id,
            "full_name": full_name,
            "embedding": _normalize_embedding(embedding),
        })
    return students


def insert_attendance_record(
    student_id: str,
    student_name: str,
    class_name: str,
    confidence: float,
    source: str = "camera",
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance_records (student_id, student_name, class_name, attendance_date, attendance_time, confidence, source)
                VALUES (%s, %s, %s, CURRENT_DATE, CURRENT_TIME, %s, %s)
                """,
                (student_id, student_name, class_name, float(confidence), source),
            )
        conn.commit()


def fetch_recent_attendance(limit: int = 50) -> List[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_id, student_name, class_name, attendance_date, attendance_time, confidence, source
                FROM attendance_records
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "student_id": row[0],
            "student_name": row[1],
            "class_name": row[2],
            "attendance_date": row[3],
            "attendance_time": row[4],
            "confidence": float(row[5] or 0.0),
            "source": row[6],
        }
        for row in rows
    ]
