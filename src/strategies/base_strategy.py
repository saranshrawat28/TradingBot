"""
Base class for all Algorithmic Trading Strategies.
Implements the Template Method pattern for reliable data validation, copying, and signal lifecycle.
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for technical trading strategies.
    Signal convention:
      1: BUY / LONG Entry
     -1: SELL / SHORT Entry
      2: EXIT LONG
     -2: EXIT SHORT
      0: HOLD / NO ACTION
    """
    
    def __init__(self, name: str, params: dict = None, min_period: int = 20):
        self.name = name
        self.params = params or {}
        self.min_period = min_period
        
    def _compute_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Subclasses override this template method to calculate indicators
        and assign 'Signal' (1, -1, 2, -2, 0) and 'Signal_Reason'.
        """
        raise NotImplementedError("Subclasses must implement _compute_signals")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Template Method: Validates DataFrame, handles boundary conditions,
        initializes output columns, and delegates to _compute_signals.
        """
        if df.empty or len(df) < self.min_period:
            out = df.copy() if not df.empty else pd.DataFrame(columns=["Close", "Signal", "Signal_Reason"])
            out["Signal"] = 0
            out["Signal_Reason"] = "Insufficient data"
            return out
            
        data = df.copy()
        if "Signal" not in data.columns:
            data["Signal"] = 0
        if "Signal_Reason" not in data.columns:
            data["Signal_Reason"] = ""
            
        try:
            return self._compute_signals(data)
        except NotImplementedError:
            # If subclass overrides generate_signals directly, return data
            return data
        
    def get_latest_signal(self, df: pd.DataFrame) -> dict:
        """Get signal for the most recent candle."""
        df_sig = self.generate_signals(df)
        if df_sig.empty:
            return {"signal": 0, "action": "HOLD", "reason": "No data"}
            
        last_row = df_sig.iloc[-1]
        sig = int(last_row.get("Signal", 0))
        
        action_map = {
            1: "BUY",
            -1: "SELL",
            2: "EXIT_LONG",
            -2: "EXIT_SHORT",
            0: "HOLD"
        }
        
        return {
            "signal": sig,
            "action": action_map.get(sig, "HOLD"),
            "price": float(last_row["Close"]),
            "timestamp": str(df_sig.index[-1]),
            "reason": str(last_row.get("Signal_Reason", "Technical Signal"))
        }
