"""
AI Confidence Calibration and Outcome Tracking Engine.
Logs raw LLM confidence scores against actual realized trade returns to empirically validate model calibration.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional
from src.utils.helpers import get_ist_now

CALIBRATION_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "confidence_calibration.jsonl")

class ConfidenceCalibrator:
    """
    Logs and analyzes the relationship between AI confidence scores and trading win-rates.
    """
    
    @staticmethod
    def log_decision(
        provider: str,
        model: str,
        symbol: str,
        action: str,
        confidence_score: float,
        reasoning: str,
        trade_id: Optional[str] = None
    ) -> None:
        """Log a new AI decision for future outcome matching."""
        os.makedirs(os.path.dirname(CALIBRATION_LOG_FILE), exist_ok=True)
        entry = {
            "timestamp": get_ist_now().isoformat(),
            "trade_id": trade_id or f"D_{int(datetime.now().timestamp())}",
            "provider": provider,
            "model": model,
            "symbol": symbol,
            "action": action,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            "realized_pnl": None,
            "is_profitable": None,
            "status": "OPEN" if action in ["BUY_CALL", "BUY_PUT", "BUY_STOCK"] else "RESOLVED"
        }
        with open(CALIBRATION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def update_trade_outcome(trade_id: str, pnl: float) -> None:
        """Update calibration log with final trade outcome."""
        if not os.path.exists(CALIBRATION_LOG_FILE):
            return
            
        entries = []
        with open(CALIBRATION_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
                        
        for e in entries:
            if e.get("trade_id") == trade_id:
                e["realized_pnl"] = pnl
                e["is_profitable"] = pnl > 0
                e["status"] = "RESOLVED"
                
        with open(CALIBRATION_LOG_FILE, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    @staticmethod
    def get_calibration_stats() -> dict:
        """
        Calculate expected vs actual accuracy by confidence bucket (e.g. 7-8, 8-9, 9-10).
        """
        if not os.path.exists(CALIBRATION_LOG_FILE):
            return {"total_decisions": 0, "win_rate": 0.0, "buckets": {}}
            
        resolved = []
        with open(CALIBRATION_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("status") == "RESOLVED" and data.get("is_profitable") is not None:
                            resolved.append(data)
                    except Exception:
                        pass
                        
        if not resolved:
            return {"total_decisions": 0, "win_rate": 0.0, "buckets": {}}
            
        wins = [r for r in resolved if r["is_profitable"]]
        overall_wr = (len(wins) / len(resolved)) * 100.0
        
        # Bucket analysis
        buckets = {"7.0 - 7.9": [], "8.0 - 8.9": [], "9.0 - 10.0": []}
        for r in resolved:
            score = r.get("confidence_score", 0.0)
            if 7.0 <= score < 8.0:
                buckets["7.0 - 7.9"].append(r["is_profitable"])
            elif 8.0 <= score < 9.0:
                buckets["8.0 - 8.9"].append(r["is_profitable"])
            elif score >= 9.0:
                buckets["9.0 - 10.0"].append(r["is_profitable"])
                
        bucket_stats = {}
        for b_name, b_results in buckets.items():
            if b_results:
                bucket_stats[b_name] = {
                    "count": len(b_results),
                    "win_rate": round((sum(b_results) / len(b_results)) * 100.0, 1)
                }
                
        return {
            "total_decisions": len(resolved),
            "win_rate": round(overall_wr, 1),
            "buckets": bucket_stats
        }
