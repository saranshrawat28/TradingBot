"""
Bollinger Bands Mean Reversion & Volatility Breakout Strategy.
"""

import pandas as pd
from src.strategies.base_strategy import BaseStrategy
from src.strategies.indicators import calculate_bollinger_bands, calculate_rsi

class BollingerBandsStrategy(BaseStrategy):
    """
    Strategy Modes:
    1. Mean Reversion: Buy when price touches lower band + RSI oversold (<35), Sell when price touches upper band + RSI overbought (>65).
    2. Breakout: Buy when price closes above upper band during high volume / bandwidth expansion.
    """
    
    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        rsi_period: int = 14,
        mode: str = "Mean Reversion" # "Mean Reversion" or "Breakout"
    ):
        super().__init__(
            name="Bollinger Bands Dynamic Strategy",
            params={
                "period": period,
                "num_std": num_std,
                "rsi_period": rsi_period,
                "mode": mode
            }
        )
        self.period = period
        self.num_std = num_std
        self.rsi_period = rsi_period
        self.mode = mode

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < self.period + 2:
            df["Signal"] = 0
            df["Signal_Reason"] = ""
            return df
            
        data = df.copy()
        close = data["Close"]
        
        data["BB_Upper"], data["BB_Middle"], data["BB_Lower"], data["BB_Width"], data["BB_PctB"] = calculate_bollinger_bands(
            close, self.period, self.num_std
        )
        data["RSI"] = calculate_rsi(close, self.rsi_period)
        
        data["Signal"] = 0
        data["Signal_Reason"] = ""
        
        if self.mode == "Mean Reversion":
            # Long: Price dipped below or near lower band & RSI < 40, now bouncing up
            prev_close = close.shift(1)
            buy_condition = (prev_close <= data["BB_Lower"]) & (close > data["BB_Lower"]) & (data["RSI"] < 45)
            # Short / Exit: Price crossed upper band & RSI > 60
            sell_condition = (prev_close >= data["BB_Upper"]) & (close < data["BB_Upper"]) & (data["RSI"] > 55)
            
            data.loc[buy_condition, "Signal"] = 1
            data.loc[buy_condition, "Signal_Reason"] = "BB Lower Band Mean Reversion Bounce + RSI Oversold"
            
            data.loc[sell_condition, "Signal"] = -1
            data.loc[sell_condition, "Signal_Reason"] = "BB Upper Band Mean Reversion Reject + RSI Overbought"
        else:
            # Breakout Mode
            buy_condition = (close > data["BB_Upper"]) & (data["BB_PctB"] > 1.0)
            sell_condition = (close < data["BB_Lower"]) & (data["BB_PctB"] < 0.0)
            
            data.loc[buy_condition, "Signal"] = 1
            data.loc[buy_condition, "Signal_Reason"] = "BB Upper Band Volatility Breakout"
            
            data.loc[sell_condition, "Signal"] = -1
            data.loc[sell_condition, "Signal_Reason"] = "BB Lower Band Breakdown"
            
        return data
