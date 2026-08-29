"""
Isolated SQLite Database Layer for ApexTrade Paper Trading Lab.
Database file: storage/paper_lab.db
"""

import sqlite3
import json
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional
import config

DB_PATH = config.STORAGE_DIR / "paper_lab.db"
_db_lock = threading.RLock()

class PaperDB:
    """Thread-safe SQLite storage engine for Paper Trading Accuracy Lab."""

    @staticmethod
    def get_connection():
        """Creates a thread-safe connection."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def init_db(cls):
        """Initializes all paper lab tables with idempotent constraints."""
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()

            # 1. Paper Picks (Daily Candidate Recommendations & Real 9:15 Fills)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_picks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pick_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                display_name TEXT NOT NULL,
                signal_time TEXT NOT NULL,
                signal_price REAL NOT NULL,
                entry_time TEXT,
                entry_price REAL,
                target_1 REAL NOT NULL,
                target_2 REAL NOT NULL,
                stop_loss REAL NOT NULL,
                allocated_capital REAL NOT NULL DEFAULT 20000.0,
                quantity INTEGER NOT NULL DEFAULT 1,
                advisor_score REAL NOT NULL,
                setup_grade TEXT,
                score_breakdown TEXT,
                top_signals TEXT,
                config_version TEXT NOT NULL,
                data_stale_flag INTEGER DEFAULT 0,
                status TEXT DEFAULT 'PENDING_OPEN',
                created_at TEXT NOT NULL,
                UNIQUE(pick_date, symbol)
            )
            """)

            # 2. Paper Outcomes (Ground-Truth EOD Replay Results)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pick_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                exit_time TEXT NOT NULL,
                exit_type TEXT NOT NULL,
                pnl_rs REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                allocated_capital REAL NOT NULL,
                quantity INTEGER NOT NULL,
                max_favorable_excursion_rs REAL DEFAULT 0.0,
                max_adverse_excursion_rs REAL DEFAULT 0.0,
                bars_held INTEGER DEFAULT 0,
                resolution_method TEXT DEFAULT '1M_CANDLE_REPLAY',
                evaluated_at TEXT NOT NULL,
                UNIQUE(pick_date, symbol)
            )
            """)

            # 3. Paper Snapshots (Intraday UI Telemetry)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pick_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl_rs REAL NOT NULL,
                unrealized_pnl_pct REAL NOT NULL,
                status TEXT NOT NULL
            )
            """)

            conn.commit()
            conn.close()

    @classmethod
    def save_pending_picks(cls, picks: List[Dict[str, Any]]) -> int:
        """
        Saves candidate signals generated at 08:50 AM with status PENDING_OPEN.
        Idempotent (skips duplicates for same day & symbol).
        """
        cls.init_db()
        inserted_count = 0
        now_str = datetime.now().isoformat()

        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()

            for p in picks:
                try:
                    cursor.execute("""
                    INSERT INTO paper_picks (
                        pick_date, symbol, display_name, signal_time, signal_price,
                        entry_time, entry_price, target_1, target_2, stop_loss,
                        allocated_capital, quantity, advisor_score, setup_grade,
                        score_breakdown, top_signals, config_version, data_stale_flag,
                        status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pick_date, symbol) DO NOTHING
                    """, (
                        p["pick_date"],
                        p["symbol"],
                        p.get("display_name", p["symbol"]),
                        p.get("signal_time", now_str),
                        float(p.get("signal_price", 0.0)),
                        p.get("entry_time"),
                        p.get("entry_price"),
                        float(p.get("target_1", 0.0)),
                        float(p.get("target_2", 0.0)),
                        float(p.get("stop_loss", 0.0)),
                        float(p.get("allocated_capital", 20000.0)),
                        int(p.get("quantity", 1)),
                        float(p.get("advisor_score", 0.0)),
                        p.get("setup_grade", "GRADE A"),
                        json.dumps(p.get("score_breakdown", {})),
                        json.dumps(p.get("top_signals", [])),
                        p.get("config_version", "v1.0.0"),
                        int(p.get("data_stale_flag", 0)),
                        p.get("status", "PENDING_OPEN"),
                        now_str
                    ))
                    if cursor.rowcount > 0:
                        inserted_count += 1
                except Exception as e:
                    print(f"[PaperDB] Error saving pick {p.get('symbol')}: {e}")

            conn.commit()
            conn.close()
        return inserted_count

    @classmethod
    def update_pick_fill(
        cls,
        pick_date: str,
        symbol: str,
        entry_price: float,
        entry_time: str,
        target_1: float,
        target_2: float,
        stop_loss: float,
        quantity: int
    ):
        """Updates pick with the real 09:15 AM open price fill and marks it ACTIVE."""
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE paper_picks
            SET entry_price = ?,
                entry_time = ?,
                target_1 = ?,
                target_2 = ?,
                stop_loss = ?,
                quantity = ?,
                status = 'ACTIVE'
            WHERE pick_date = ? AND symbol = ?
            """, (entry_price, entry_time, target_1, target_2, stop_loss, quantity, pick_date, symbol))
            conn.commit()
            conn.close()

    @classmethod
    def get_picks_by_date(cls, pick_date: str) -> List[Dict[str, Any]]:
        """Returns all picks recorded for a specific date."""
        cls.init_db()
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM paper_picks WHERE pick_date = ? ORDER BY advisor_score DESC", (pick_date,))
            rows = cursor.fetchall()
            conn.close()

            results = []
            for r in rows:
                d = dict(r)
                if d.get("score_breakdown"):
                    try: d["score_breakdown"] = json.loads(d["score_breakdown"])
                    except: pass
                if d.get("top_signals"):
                    try: d["top_signals"] = json.loads(d["top_signals"])
                    except: pass
                results.append(d)
            return results

    @classmethod
    def get_pending_picks(cls, pick_date: str) -> List[Dict[str, Any]]:
        """Returns picks that are waiting for market open fill."""
        cls.init_db()
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM paper_picks WHERE pick_date = ? AND status = 'PENDING_OPEN'", (pick_date,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]

    @classmethod
    def get_active_picks_unresolved(cls, pick_date: str) -> List[Dict[str, Any]]:
        """Returns picks for today that have not been evaluated in paper_outcomes yet."""
        cls.init_db()
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            SELECT p.* FROM paper_picks p
            LEFT JOIN paper_outcomes o ON p.pick_date = o.pick_date AND p.symbol = o.symbol
            WHERE p.pick_date = ? AND o.id IS NULL
            """, (pick_date,))
            rows = cursor.fetchall()
            conn.close()

            results = []
            for r in rows:
                d = dict(r)
                if d.get("score_breakdown"):
                    try: d["score_breakdown"] = json.loads(d["score_breakdown"])
                    except: pass
                results.append(d)
            return results

    @classmethod
    def save_outcome(cls, outcome: Dict[str, Any]):
        """Saves final chronological evaluation outcome for a pick."""
        cls.init_db()
        now_str = datetime.now().isoformat()

        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO paper_outcomes (
                pick_date, symbol, entry_price, exit_price, exit_time,
                exit_type, pnl_rs, pnl_pct, allocated_capital, quantity,
                max_favorable_excursion_rs, max_adverse_excursion_rs,
                bars_held, resolution_method, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pick_date, symbol) DO UPDATE SET
                entry_price = excluded.entry_price,
                exit_price = excluded.exit_price,
                exit_time = excluded.exit_time,
                exit_type = excluded.exit_type,
                pnl_rs = excluded.pnl_rs,
                pnl_pct = excluded.pnl_pct,
                allocated_capital = excluded.allocated_capital,
                quantity = excluded.quantity,
                max_favorable_excursion_rs = excluded.max_favorable_excursion_rs,
                max_adverse_excursion_rs = excluded.max_adverse_excursion_rs,
                bars_held = excluded.bars_held,
                resolution_method = excluded.resolution_method,
                evaluated_at = excluded.evaluated_at
            """, (
                outcome["pick_date"],
                outcome["symbol"],
                float(outcome["entry_price"]),
                float(outcome["exit_price"]),
                outcome["exit_time"],
                outcome["exit_type"],
                float(outcome["pnl_rs"]),
                float(outcome["pnl_pct"]),
                float(outcome.get("allocated_capital", 20000.0)),
                int(outcome.get("quantity", 1)),
                float(outcome.get("max_favorable_excursion_rs", 0.0)),
                float(outcome.get("max_adverse_excursion_rs", 0.0)),
                int(outcome.get("bars_held", 0)),
                outcome.get("resolution_method", "1M_CANDLE_REPLAY"),
                now_str
            ))

            # Also update status in paper_picks to CLOSED
            cursor.execute("""
            UPDATE paper_picks SET status = 'CLOSED'
            WHERE pick_date = ? AND symbol = ?
            """, (outcome["pick_date"], outcome["symbol"]))

            conn.commit()
            conn.close()

    @classmethod
    def get_outcomes_for_range(cls, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetches combined picks & outcomes data for a date range (e.g. for weekly reporting).
        """
        cls.init_db()
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            SELECT
                p.pick_date, p.symbol, p.display_name, p.advisor_score,
                p.score_breakdown, p.top_signals, p.config_version,
                p.signal_time, p.entry_time, p.data_stale_flag,
                o.entry_price, o.exit_price, o.exit_time, o.exit_type,
                o.pnl_rs, o.pnl_pct, o.allocated_capital, o.quantity,
                o.max_favorable_excursion_rs, o.max_adverse_excursion_rs,
                o.bars_held, o.resolution_method
            FROM paper_picks p
            INNER JOIN paper_outcomes o ON p.pick_date = o.pick_date AND p.symbol = o.symbol
            WHERE p.pick_date >= ? AND p.pick_date <= ?
            ORDER BY p.pick_date DESC, p.advisor_score DESC
            """, (start_date, end_date))
            rows = cursor.fetchall()
            conn.close()

            results = []
            for r in rows:
                d = dict(r)
                if d.get("score_breakdown"):
                    try: d["score_breakdown"] = json.loads(d["score_breakdown"])
                    except: pass
                if d.get("top_signals"):
                    try: d["top_signals"] = json.loads(d["top_signals"])
                    except: pass
                results.append(d)
            return results

    @classmethod
    def save_snapshot(cls, snapshot: Dict[str, Any]):
        """Saves a 30-minute intraday UI telemetry snapshot."""
        cls.init_db()
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO paper_snapshots (
                pick_date, symbol, timestamp, current_price,
                unrealized_pnl_rs, unrealized_pnl_pct, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot["pick_date"],
                snapshot["symbol"],
                snapshot["timestamp"],
                float(snapshot["current_price"]),
                float(snapshot["unrealized_pnl_rs"]),
                float(snapshot["unrealized_pnl_pct"]),
                snapshot["status"]
            ))
            conn.commit()
            conn.close()

    @classmethod
    def get_latest_snapshots(cls, pick_date: str) -> List[Dict[str, Any]]:
        """Returns the most recent snapshot for each symbol on a date."""
        cls.init_db()
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
            SELECT s.* FROM paper_snapshots s
            INNER JOIN (
                SELECT symbol, MAX(id) as max_id
                FROM paper_snapshots
                WHERE pick_date = ?
                GROUP BY symbol
            ) m ON s.id = m.max_id
            """, (pick_date,))
            rows = cursor.fetchall()
            conn.close()
            return [dict(r) for r in rows]

    @classmethod
    def get_all_dates(cls) -> List[str]:
        """Returns all distinct pick dates available in DB."""
        cls.init_db()
        with _db_lock:
            conn = cls.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT pick_date FROM paper_picks ORDER BY pick_date DESC")
            rows = cursor.fetchall()
            conn.close()
            return [r[0] for r in rows]
