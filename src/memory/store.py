from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_profile (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visit_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  category TEXT,
  visited_at TEXT NOT NULL,
  source TEXT
);
CREATE INDEX IF NOT EXISTS visit_history_visited_at_idx
  ON visit_history(visited_at);
"""


class MemoryStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- user_profile ---
    def set_user_profile(self, key: str, value: Any) -> None:
        now = datetime.now().isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO user_profile(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), now),
            )

    def get_user_profile(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            row = c.execute(
                "SELECT value FROM user_profile WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    def all_user_profile(self) -> dict[str, Any]:
        with self._conn() as c:
            rows = c.execute("SELECT key,value FROM user_profile").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def remove_user_profile(self, key: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM user_profile WHERE key=?", (key,))

    def clear_all(self) -> None:
        """profile + visit_history 둘 다 비움."""
        with self._conn() as c:
            c.execute("DELETE FROM user_profile")
            c.execute("DELETE FROM visit_history")

    # --- visit_history ---
    def append_visit(
        self,
        name: str,
        category: str | None,
        visited_at: datetime | None = None,
        source: str = "recommended",
    ) -> None:
        ts = (visited_at or datetime.now()).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT INTO visit_history(name,category,visited_at,source) VALUES(?,?,?,?)",
                (name, category, ts, source),
            )

    def get_recent_visits(
        self, within_days: int, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        ref = (now or datetime.now()).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT name,category,visited_at,source FROM visit_history "
                "WHERE julianday(?) - julianday(visited_at) <= ? "
                "ORDER BY visited_at DESC",
                (ref, within_days),
            ).fetchall()
        return [dict(r) for r in rows]
