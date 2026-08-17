"""
SuperTrend Trend-Following Strategy (Standard Indian Intraday & Swing Strategy).
"""

import pandas as pd
from src.strategies.base_strategy import BaseStrategy
from src.strategies.indicators import calculate_supertrend, calculate_ema

class SuperTrendStrategy(BaseStrategy):
    """
    Strategy Rules:
    - BUY: SuperTrend flips to GREEN / BULLISH (+1) (and optionally above 200 EMA).
    - SELL: SuperTrend flips to RED / BEARISH (-1).
    """
    
    def __init__(
        self,
        atr_period: int = 10,
        multiplier: float = 3.0,
        trend_ema: int = 50,
        use_ema_filter: bool = True
    ):
        super().__init__(
            name="SuperTrend ATR Dynamic Trend",
            params={
                "atr_period": atr_period,
                "multiplier": multiplier,
                "trend_ema": trend_ema,
                "use_ema_filter": use_ema_filter
            }
        )
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.trend_ema = trend_ema
        self.use_ema_filter = use_ema_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < self.atr_period + 5:
            df["Signal"] = 0
            df["Signal_Reason"] = ""
            return df
            
        data = df.copy()
        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        
        data["SuperTrend"], data["SuperTrend_Dir"] = calculate_supertrend(
            high, low, close, self.atr_period, self.multiplier
        )
        data["EMA_Filter"] = calculate_ema(close, self.trend_ema)
        
        prev_dir = data["SuperTrend_Dir"].shift(1)
        curr_dir = data["SuperTrend_Dir"]
        
        flip_bullish = (prev_dir == -1) & (curr_dir == 1)
        flip_bearish = (prev_dir == 1) & (curr_dir == -1)
        
        if self.use_ema_filter:
            flip_bullish = flip_bullish & (close >= data["EMA_Filter"])
            flip_bearish = flip_bearish & (close <= data["EMA_Filter"])
            
        data["Signal"] = 0
        data["Signal_Reason"] = ""
        
        data.loc[flip_bullish, "Signal"] = 1
        data.loc[flip_bullish, "Signal_Reason"] = f"SuperTrend Bullish Flip (ATR {self.atr_period}, Mult {self.multiplier})"
        
        data.loc[flip_bearish, "Signal"] = -1
        data.loc[flip_bearish, "Signal_Reason"] = f"SuperTrend Bearish Flip (ATR {self.atr_period}, Mult {self.multiplier})"
        
        return data
