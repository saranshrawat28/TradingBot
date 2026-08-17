"""
Vectorized Technical Indicator calculations using pure NumPy & Pandas.
Rock-solid reliability with zero external C-compilation dependencies.
"""

import pandas as pd
import numpy as np

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    return series.rolling(window=period, min_periods=1).mean()

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) using Wilder's exponential smoothing.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's Exponential Smoothing (alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    Returns (macd_line, signal_line, macd_histogram).
    """
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    Returns (upper_band, middle_band, lower_band, bandwidth, percent_b).
    """
    middle_band = calculate_sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=1).std()
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    bandwidth = ((upper_band - lower_band) / middle_band.replace(0, np.nan)) * 100
    percent_b = (series - lower_band) / (upper_band - lower_band).replace(0, np.nan)
    return upper_band, middle_band, lower_band, bandwidth.fillna(0), percent_b.fillna(0.5)

def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr.fillna(tr.mean())

def calculate_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate SuperTrend indicator.
    Returns (supertrend_line, trend_direction: 1 for Bullish, -1 for Bearish).
    """
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)
    
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    trend = pd.Series(1, index=close.index)
    supertrend = pd.Series(np.nan, index=close.index)
    
    for i in range(1, len(close)):
        # Upper band adjustment
        if upper_basic.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]
            
        # Lower band adjustment
        if lower_basic.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]
            
        # Trend switch
        if trend.iloc[i-1] == 1:
            if close.iloc[i] < lower_band.iloc[i]:
                trend.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                trend.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
        else:
            if close.iloc[i] > upper_band.iloc[i]:
                trend.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                trend.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
                
    # Fill first value
    if len(supertrend) > 0 and pd.isna(supertrend.iloc[0]):
        supertrend.iloc[0] = lower_band.iloc[0]
        
    return supertrend, trend

def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)."""
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = pv.cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).fillna(close)

def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX), +DI, and -DI using Wilder's smoothing.
    Returns (adx, plus_di, minus_di).
    """
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm_series = pd.Series(plus_dm, index=high.index)
    minus_dm_series = pd.Series(minus_dm, index=low.index)
    
    atr = calculate_atr(high, low, close, period)
    
    smooth_plus_dm = plus_dm_series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm_series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    plus_di = (smooth_plus_dm / atr.replace(0, np.nan)) * 100
    minus_di = (smooth_minus_dm / atr.replace(0, np.nan)) * 100
    
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20.0), plus_di.fillna(20.0), minus_di.fillna(20.0)

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard set of indicators to candle DataFrame."""
    if df.empty or len(df) < 5:
        return df
        
    res = df.copy()
    close = res["Close"]
    high = res["High"]
    low = res["Low"]
    volume = res["Volume"] if "Volume" in res.columns else pd.Series(1, index=res.index)
    
    # EMAs
    res["EMA_9"] = calculate_ema(close, 9)
    res["EMA_21"] = calculate_ema(close, 21)
    res["EMA_50"] = calculate_ema(close, 50)
    res["EMA_200"] = calculate_ema(close, 200)
    
    # RSI
    res["RSI_14"] = calculate_rsi(close, 14)
    
    # MACD
    res["MACD"], res["MACD_Signal"], res["MACD_Hist"] = calculate_macd(close, 12, 26, 9)
    
    # Bollinger Bands
    res["BB_Upper"], res["BB_Middle"], res["BB_Lower"], res["BB_Bandwidth"], res["BB_PctB"] = calculate_bollinger_bands(close, 20, 2.0)
    
    # ATR & SuperTrend
    res["ATR_14"] = calculate_atr(high, low, close, 14)
    res["SuperTrend"], res["SuperTrend_Dir"] = calculate_supertrend(high, low, close, 10, 3.0)
    
    # ADX & Directional Movement (Regime Filter)
    res["ADX_14"], res["Plus_DI"], res["Minus_DI"] = calculate_adx(high, low, close, 14)
    
    # VWAP
    res["VWAP"] = calculate_vwap(high, low, close, volume)
    
    return res
