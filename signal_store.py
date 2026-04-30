import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from strategy import Signal


class SignalStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sent_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_date TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sent_signals_date
                ON sent_signals(signal_date)
                """
            )
            conn.commit()

    def today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def count_today(self) -> int:
        today = self.today_key()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM sent_signals WHERE signal_date = ?",
                (today,),
            ).fetchone()
        return int(row["cnt"])

    def can_send_today(self, max_signals_per_day: int) -> bool:
        return self.count_today() < max_signals_per_day

    def _fingerprint(self, signal: Signal) -> str:
        payload = f"{self.today_key()}|{signal.pair}|{signal.side}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_duplicate(self, signal: Signal) -> bool:
        fp = self._fingerprint(signal)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sent_signals WHERE fingerprint = ? LIMIT 1",
                (fp,),
            ).fetchone()
        return row is not None

    def save(self, signal: Signal) -> None:
        fp = self._fingerprint(signal)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sent_signals (
                    signal_date, pair, side, confidence, entry, stop_loss,
                    take_profit, fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.today_key(),
                    signal.pair,
                    signal.side,
                    signal.confidence,
                    signal.entry,
                    signal.stop_loss,
                    signal.take_profit,
                    fp,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
