"""
Quantitative Research Experiment Journal & Strategy Audit Store.
Persists hypotheses, walk-forward performance metrics, and model configurations to SQLite.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import config
from src.utils.storage import get_connection

class ResearchJournal:
    """
    Tracks and documents all quantitative trading experiments, hypothesis tests, and out-of-sample metrics.
    """
    
    @staticmethod
    def init_table():
        """Initialize research_experiments table in database."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            model_type TEXT NOT NULL,
            oos_sharpe REAL NOT NULL,
            deflated_sharpe REAL NOT NULL,
            oos_cagr REAL NOT NULL,
            oos_max_dd REAL NOT NULL,
            win_rate REAL NOT NULL,
            consistency_pct REAL NOT NULL,
            notes TEXT
        )
        """)
        conn.commit()
        conn.close()
        
    @staticmethod
    def log_experiment(
        symbol: str,
        hypothesis: str,
        model_type: str,
        oos_sharpe: float,
        deflated_sharpe: float,
        oos_cagr: float,
        oos_max_dd: float,
        win_rate: float,
        consistency_pct: float = 100.0,
        notes: str = ""
    ) -> int:
        """Log a systematic backtest run to the research journal."""
        ResearchJournal.init_table()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO research_experiments (
            timestamp, symbol, hypothesis, model_type, oos_sharpe, deflated_sharpe, oos_cagr, oos_max_dd, win_rate, consistency_pct, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            hypothesis,
            model_type,
            oos_sharpe,
            deflated_sharpe,
            oos_cagr,
            oos_max_dd,
            win_rate,
            consistency_pct,
            notes
        ))
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    @staticmethod
    def get_experiments(limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent research experiments."""
        ResearchJournal.init_table()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM research_experiments ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

# Auto-initialize on import
ResearchJournal.init_table()
