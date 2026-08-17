"""
Multi-Indicator Confluence Strategy (Institutional Grade).
Combines Trend (EMA), Momentum (RSI + MACD), and Volatility (SuperTrend) filters.
"""

import pandas as pd
from src.strategies.base_strategy import BaseStrategy
from src.strategies.indicators import (
    calculate_ema, calculate_rsi, calculate_macd, calculate_supertrend
)

class MultiIndicatorConfluenceStrategy(BaseStrategy):
    """
    High-probability confluence setup:
    - Long Entry: 9 EMA > 21 EMA + RSI between 52 and 68 + MACD Histogram > 0 + SuperTrend Bullish.
    - Short / Exit: 9 EMA < 21 EMA + RSI < 48 + MACD Histogram < 0 + SuperTrend Bearish.
    """
    
    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        rsi_period: int = 14,
        supertrend_atr: int = 10,
        supertrend_mult: float = 3.0
    ):
        super().__init__(
            name="Multi-Indicator Confluence (Trend + Momentum + Volatility)",
            params={
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "rsi_period": rsi_period,
                "supertrend_atr": supertrend_atr,
                "supertrend_mult": supertrend_mult
            }
        )
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_period = rsi_period
        self.supertrend_atr = supertrend_atr
        self.supertrend_mult = supertrend_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < max(self.slow_ema, self.rsi_period) + 10:
            df["Signal"] = 0
            df["Signal_Reason"] = ""
            return df
            
        data = df.copy()
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        
        # Calculate Indicators
        data["EMA_Fast"] = calculate_ema(close, self.fast_ema)
        data["EMA_Slow"] = calculate_ema(close, self.slow_ema)
        data["RSI"] = calculate_rsi(close, self.rsi_period)
        data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = calculate_macd(close, 12, 26, 9)
        data["SuperTrend"], data["SuperTrend_Dir"] = calculate_supertrend(
            high, low, close, self.supertrend_atr, self.supertrend_mult
        )
        
        # Conditions
        trend_bull = (data["EMA_Fast"] > data["EMA_Slow"]) & (data["SuperTrend_Dir"] == 1)
        momentum_bull = (data["RSI"] >= 50) & (data["RSI"] <= 72) & (data["MACD_Hist"] > 0)
        
        trend_bear = (data["EMA_Fast"] < data["EMA_Slow"]) & (data["SuperTrend_Dir"] == -1)
        momentum_bear = (data["RSI"] <= 50) & (data["RSI"] >= 28) & (data["MACD_Hist"] < 0)
        
        # Trigger on transitions or fresh breakouts
        prev_trend_bull = (data["EMA_Fast"].shift(1) > data["EMA_Slow"].shift(1)) & (data["SuperTrend_Dir"].shift(1) == 1)
        fresh_bull = trend_bull & momentum_bull & (~prev_trend_bull | (data["MACD_Hist"].shift(1) <= 0))
        
        prev_trend_bear = (data["EMA_Fast"].shift(1) < data["EMA_Slow"].shift(1)) & (data["SuperTrend_Dir"].shift(1) == -1)
        fresh_bear = trend_bear & momentum_bear & (~prev_trend_bear | (data["MACD_Hist"].shift(1) >= 0))
        
        data["Signal"] = 0
        data["Signal_Reason"] = ""
        
        data.loc[fresh_bull, "Signal"] = 1
        data.loc[fresh_bull, "Signal_Reason"] = "Multi-Confluence: Bullish EMA + RSI > 50 + MACD > 0 + SuperTrend Green"
        
        data.loc[fresh_bear, "Signal"] = -1
        data.loc[fresh_bear, "Signal_Reason"] = "Multi-Confluence: Bearish EMA + RSI < 50 + MACD < 0 + SuperTrend Red"
        
        return data
