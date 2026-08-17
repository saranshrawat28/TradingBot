"""
EMA Crossover with RSI Momentum Confirmation Strategy.
Popular, highly-effective intraday and swing strategy for Indian Equities.
"""

import pandas as pd
import numpy as np
from src.strategies.base_strategy import BaseStrategy
from src.strategies.indicators import calculate_ema, calculate_rsi

class EmaRsiStrategy(BaseStrategy):
    """
    Strategy Rules:
    - BUY: Fast EMA crosses ABOVE Slow EMA AND RSI > rsi_buy_threshold (and optionally Close > 200 EMA).
    - SELL / SHORT: Fast EMA crosses BELOW Slow EMA AND RSI < rsi_sell_threshold.
    - EXIT: Opposite EMA cross or RSI overbought/oversold reversal.
    """
    
    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        trend_ema: int = 200,
        rsi_period: int = 14,
        rsi_buy_threshold: float = 50.0,
        rsi_sell_threshold: float = 50.0,
        use_trend_filter: bool = True
    ):
        super().__init__(
            name="EMA Crossover + RSI Momentum",
            params={
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "trend_ema": trend_ema,
                "rsi_period": rsi_period,
                "rsi_buy_threshold": rsi_buy_threshold,
                "rsi_sell_threshold": rsi_sell_threshold,
                "use_trend_filter": use_trend_filter
            }
        )
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.trend_ema = trend_ema
        self.rsi_period = rsi_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.use_trend_filter = use_trend_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < max(self.slow_ema, self.rsi_period) + 2:
            df["Signal"] = 0
            df["Signal_Reason"] = ""
            return df
            
        data = df.copy()
        close = data["Close"]
        
        # Calculate Indicators
        data["EMA_Fast"] = calculate_ema(close, self.fast_ema)
        data["EMA_Slow"] = calculate_ema(close, self.slow_ema)
        data["EMA_Trend"] = calculate_ema(close, self.trend_ema)
        data["RSI"] = calculate_rsi(close, self.rsi_period)
        
        # Crossover Detection
        prev_fast = data["EMA_Fast"].shift(1)
        prev_slow = data["EMA_Slow"].shift(1)
        
        bullish_cross = (prev_fast <= prev_slow) & (data["EMA_Fast"] > data["EMA_Slow"])
        bearish_cross = (prev_fast >= prev_slow) & (data["EMA_Fast"] < data["EMA_Slow"])
        
        # RSI Filters
        rsi_bullish = data["RSI"] >= self.rsi_buy_threshold
        rsi_bearish = data["RSI"] <= self.rsi_sell_threshold
        
        # Trend Filter (Close above 200 EMA)
        trend_bullish = (close >= data["EMA_Trend"]) if self.use_trend_filter else True
        trend_bearish = (close <= data["EMA_Trend"]) if self.use_trend_filter else True
        
        data["Signal"] = 0
        data["Signal_Reason"] = ""
        
        # Long Entry
        buy_condition = bullish_cross & rsi_bullish & trend_bullish
        data.loc[buy_condition, "Signal"] = 1
        data.loc[buy_condition, "Signal_Reason"] = f"Bullish Cross ({self.fast_ema}/{self.slow_ema} EMA) + RSI > {self.rsi_buy_threshold}"
        
        # Short / Sell Entry
        sell_condition = bearish_cross & rsi_bearish & trend_bearish
        data.loc[sell_condition, "Signal"] = -1
        data.loc[sell_condition, "Signal_Reason"] = f"Bearish Cross ({self.fast_ema}/{self.slow_ema} EMA) + RSI < {self.rsi_sell_threshold}"
        
        return data
