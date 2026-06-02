from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.db import db_cursor


def main():
    schema = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
    with db_cursor(commit=True) as cur:
        cur.execute(schema.read_text(encoding="utf-8"))
    print(f"Applied {schema}")


if __name__ == "__main__":
    main()
