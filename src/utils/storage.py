"""
SQLite Storage engine for persisting trades, active positions, portfolio balance, and bot state.
Features thread-safe reentrant locks (RLock) to prevent 'database is locked' errors during concurrent scans.
"""

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path
import config
from src.utils.helpers import get_ist_now

DB_PATH = config.STORAGE_DIR / "trading_bot.db"
_db_lock = threading.RLock()

def get_connection():
    """Create a database connection to SQLite."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables if they do not exist."""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Orders table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            value REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL,
            sl REAL,
            tp REAL,
            strategy TEXT,
            broker TEXT NOT NULL,
            notes TEXT
        )
        """)
        
        # Positions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            side TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            current_price REAL NOT NULL,
            sl REAL,
            tp REAL,
            trailing_sl REAL,
            highest_price REAL,
            strategy TEXT,
            unrealized_pnl REAL DEFAULT 0.0,
            unrealized_pnl_pct REAL DEFAULT 0.0,
            target_1 REAL,
            target_2 REAL,
            target_1_hit INTEGER DEFAULT 0,
            stage TEXT DEFAULT 'ACTIVE',
            atr REAL DEFAULT 0.0,
            initial_risk_r REAL DEFAULT 0.0,
            locked_r REAL DEFAULT 0.0
        )
        """)
        
        # Auto-migration for existing databases
        for col_def in [
            "ALTER TABLE positions ADD COLUMN target_1 REAL",
            "ALTER TABLE positions ADD COLUMN target_2 REAL",
            "ALTER TABLE positions ADD COLUMN target_1_hit INTEGER DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN stage TEXT DEFAULT 'ACTIVE'",
            "ALTER TABLE positions ADD COLUMN atr REAL DEFAULT 0.0",
            "ALTER TABLE positions ADD COLUMN initial_risk_r REAL DEFAULT 0.0",
            "ALTER TABLE positions ADD COLUMN locked_r REAL DEFAULT 0.0"
        ]:
            try:
                cursor.execute(col_def)
            except Exception:
                pass
        
        # Closed Trades History (Trade Journal)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS closed_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            gross_pnl REAL NOT NULL,
            net_pnl REAL NOT NULL,
            pnl_pct REAL NOT NULL,
            exit_reason TEXT,
            strategy TEXT,
            broker TEXT NOT NULL
        )
        """)

        # Calibration & AI Divergence Journal
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibration_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            math_score REAL NOT NULL,
            llm_confidence REAL NOT NULL,
            market_regime TEXT NOT NULL,
            proposed_action TEXT NOT NULL,
            final_action TEXT NOT NULL,
            disagreement INTEGER DEFAULT 0,
            disagreement_reason TEXT,
            entry_price REAL,
            outcome TEXT,
            pnl_pct REAL,
            prompt_version TEXT DEFAULT 'v2.0',
            model_id TEXT
        )
        """)
        
        # Portfolio Balance & State
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY,
            initial_capital REAL NOT NULL,
            cash REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            updated_at TEXT NOT NULL
        )
        """)
        
        # Insert default initial portfolio state if table empty
        cursor.execute("SELECT COUNT(*) as count FROM portfolio_state")
        if cursor.fetchone()["count"] == 0:
            cursor.execute(
                "INSERT INTO portfolio_state (id, initial_capital, cash, realized_pnl, updated_at) VALUES (1, ?, ?, 0.0, ?)",
                (config.DEFAULT_INITIAL_CAPITAL, config.DEFAULT_INITIAL_CAPITAL, datetime.now().isoformat())
            )
            
        conn.commit()
        conn.close()

# ----------------- DB Operations -----------------

def log_order(order_dict: dict) -> int:
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO orders (timestamp, symbol, side, order_type, price, quantity, value, fee, status, sl, tp, strategy, broker, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_dict.get("timestamp", datetime.now().isoformat()),
            order_dict["symbol"],
            order_dict["side"],
            order_dict.get("order_type", "MARKET"),
            order_dict["price"],
            order_dict["quantity"],
            order_dict.get("value", order_dict["price"] * order_dict["quantity"]),
            order_dict.get("fee", 0.0),
            order_dict.get("status", "FILLED"),
            order_dict.get("sl"),
            order_dict.get("tp"),
            order_dict.get("strategy", "Manual"),
            order_dict.get("broker", "paper"),
            order_dict.get("notes", "")
        ))
        order_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return order_id

def get_orders(limit: int = 50) -> list[dict]:
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

def get_open_positions() -> list[dict]:
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM positions")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

def save_position(pos: dict):
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO positions (symbol, side, entry_time, entry_price, quantity, current_price, sl, tp, trailing_sl, highest_price, strategy, unrealized_pnl, unrealized_pnl_pct, target_1, target_2, target_1_hit, stage, atr, initial_risk_r, locked_r)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            quantity=excluded.quantity,
            current_price=excluded.current_price,
            sl=excluded.sl,
            tp=excluded.tp,
            trailing_sl=excluded.trailing_sl,
            highest_price=excluded.highest_price,
            unrealized_pnl=excluded.unrealized_pnl,
            unrealized_pnl_pct=excluded.unrealized_pnl_pct,
            target_1=excluded.target_1,
            target_2=excluded.target_2,
            target_1_hit=excluded.target_1_hit,
            stage=excluded.stage,
            atr=excluded.atr,
            initial_risk_r=excluded.initial_risk_r,
            locked_r=excluded.locked_r
        """, (
            pos["symbol"], pos.get("side", "BUY"), pos.get("entry_time", get_ist_now().strftime("%Y-%m-%d %H:%M:%S")),
            pos["entry_price"], pos["quantity"],
            pos.get("current_price", pos["entry_price"]), pos.get("sl"), pos.get("tp"),
            pos.get("trailing_sl"), pos.get("highest_price", pos["entry_price"]),
            pos.get("strategy"), pos.get("unrealized_pnl", 0.0), pos.get("unrealized_pnl_pct", 0.0),
            pos.get("target_1"), pos.get("target_2"), 1 if pos.get("target_1_hit") else 0, pos.get("stage", "ACTIVE"),
            float(pos.get("atr", 0.0) or 0.0),
            float(pos.get("initial_risk_r", 0.0) or 0.0),
            float(pos.get("locked_r", 0.0) or 0.0)
        ))
        conn.commit()
        conn.close()

def delete_position(symbol: str):
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()

def clear_all_positions():
    """Clears all open positions from database."""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions")
        conn.commit()
        conn.close()

def log_closed_trade(trade: dict):
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO closed_trades (symbol, side, entry_time, exit_time, entry_price, exit_price, quantity, gross_pnl, net_pnl, pnl_pct, exit_reason, strategy, broker)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["symbol"], trade["side"], trade["entry_time"], trade["exit_time"],
            trade["entry_price"], trade["exit_price"], trade["quantity"],
            trade["gross_pnl"], trade["net_pnl"], trade["pnl_pct"],
            trade.get("exit_reason", "MANUAL"), trade.get("strategy", "MANUAL"), trade.get("broker", "paper")
        ))
        conn.commit()
        conn.close()

def get_closed_trades(limit: int = 100) -> list[dict]:
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM closed_trades ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

def get_portfolio_state() -> dict:
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM portfolio_state WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return {"initial_capital": config.DEFAULT_INITIAL_CAPITAL, "cash": config.DEFAULT_INITIAL_CAPITAL, "realized_pnl": 0.0}

def update_portfolio_state(cash: float, realized_pnl: float):
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE portfolio_state SET cash = ?, realized_pnl = ?, updated_at = ? WHERE id = 1",
            (cash, realized_pnl, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

def reset_all_data(initial_capital: float = None):
    """Reset all database tables to start fresh."""
    with _db_lock:
        cap = initial_capital if initial_capital is not None else config.DEFAULT_INITIAL_CAPITAL
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM orders")
        cursor.execute("DELETE FROM positions")
        cursor.execute("DELETE FROM closed_trades")
        cursor.execute(
            "UPDATE portfolio_state SET initial_capital = ?, cash = ?, realized_pnl = 0.0, updated_at = ? WHERE id = 1",
            (cap, cap, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

def save_ai_settings(settings: dict) -> bool:
    """Persist AI model choice, provider, and API key locally."""
    with _db_lock:
        try:
            settings_file = config.DATA_DIR / "ai_settings.json"
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception:
            return False

def load_ai_settings() -> dict:
    """Load persisted AI model choice, provider, and API key."""
    with _db_lock:
        try:
            settings_file = config.DATA_DIR / "ai_settings.json"
            if settings_file.exists():
                with open(settings_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "provider": "gemini",
            "model": "gemini-3.7-flash",
            "api_key": "",
            "is_connected": False,
            "target_asset": "NIFTY",
            "max_daily_loss": 2000.0,
            "min_confidence": 7.5
        }

def log_calibration_entry(entry: dict) -> int:
    """Log a complete decision and calibration record to SQLite."""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO calibration_journal (timestamp, symbol, math_score, llm_confidence, market_regime, proposed_action, final_action, disagreement, disagreement_reason, entry_price, outcome, pnl_pct, prompt_version, model_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get("timestamp", datetime.now().isoformat()),
            entry["symbol"],
            entry.get("math_score", 5.0),
            entry.get("llm_confidence", 5.0),
            entry.get("market_regime", "UNKNOWN"),
            entry.get("proposed_action", "HOLD"),
            entry.get("final_action", "HOLD"),
            1 if entry.get("disagreement") else 0,
            entry.get("disagreement_reason", ""),
            entry.get("entry_price"),
            entry.get("outcome", "PENDING"),
            entry.get("pnl_pct", 0.0),
            entry.get("prompt_version", "v2.0"),
            entry.get("model_id", "default")
        ))
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

def get_calibration_records(limit: int = 100) -> list[dict]:
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM calibration_journal ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

def get_disagreement_records(limit: int = 50) -> list[dict]:
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM calibration_journal WHERE disagreement = 1 ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

# Auto-initialize DB on import
init_db()
