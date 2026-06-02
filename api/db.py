from __future__ import annotations

from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector

from .config import settings


DB_CONFIG = {
    "host": settings.db_host,
    "port": settings.db_port,
    "dbname": settings.db_name,
    "user": settings.db_user,
    "password": settings.db_password,
}


@contextmanager
def db_cursor(commit: bool = False):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
