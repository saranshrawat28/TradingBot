"""
MACD (Moving Average Convergence Divergence) Momentum Strategy.
"""

import pandas as pd
from src.strategies.base_strategy import BaseStrategy
from src.strategies.indicators import calculate_macd, calculate_ema

class MacdStrategy(BaseStrategy):
    """
    Strategy Rules:
    - BUY: MACD Line crosses ABOVE Signal Line AND Histogram > 0 (or MACD crosses above 0).
    - SELL: MACD Line crosses BELOW Signal Line AND Histogram < 0.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        require_zero_cross: bool = False
    ):
        super().__init__(
            name="MACD Momentum Reversal",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "require_zero_cross": require_zero_cross
            }
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.require_zero_cross = require_zero_cross

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < self.slow_period + self.signal_period:
            df["Signal"] = 0
            df["Signal_Reason"] = ""
            return df
            
        data = df.copy()
        close = data["Close"]
        
        data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = calculate_macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )
        
        prev_macd = data["MACD"].shift(1)
        prev_signal = data["MACD_Signal"].shift(1)
        
        bullish_cross = (prev_macd <= prev_signal) & (data["MACD"] > data["MACD_Signal"])
        bearish_cross = (prev_macd >= prev_signal) & (data["MACD"] < data["MACD_Signal"])
        
        if self.require_zero_cross:
            bullish_cross = bullish_cross & (data["MACD"] > 0)
            bearish_cross = bearish_cross & (data["MACD"] < 0)
            
        data["Signal"] = 0
        data["Signal_Reason"] = ""
        
        data.loc[bullish_cross, "Signal"] = 1
        data.loc[bullish_cross, "Signal_Reason"] = "MACD Bullish Cross over Signal Line"
        
        data.loc[bearish_cross, "Signal"] = -1
        data.loc[bearish_cross, "Signal_Reason"] = "MACD Bearish Cross below Signal Line"
        
        return data
