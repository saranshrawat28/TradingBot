"""
Base class for all Algorithmic Trading Strategies.
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
    
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes OHLCV DataFrame, computes indicators, and appends a 'Signal' column.
        Returns the updated DataFrame.
        """
        pass
        
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
