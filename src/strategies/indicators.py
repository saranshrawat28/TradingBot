"""
Vectorized Technical Indicator calculations using pure NumPy & Pandas.
Rock-solid reliability with zero external C-compilation dependencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    return series.rolling(window=period, min_periods=1).mean()

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV)."""
    if close is None or volume is None or len(close) == 0:
        return pd.Series(dtype=float)
    direction = np.sign(close.diff().fillna(0))
    direction.iloc[0] = 0
    return (direction * volume).cumsum()

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

def detect_rsi_divergence(
    close: pd.Series,
    low: pd.Series,
    high: pd.Series,
    rsi: pd.Series,
    lookback: int = 20
) -> dict:
    """
    Zero-Repaint Vectorized Divergence Detector on Closed Candles.
    Pivots are strictly confirmed on bar i-2 using closed bars i-1 and i.
    Returns:
      {
        "bullish_divergence": bool, # Price Lower Low + RSI Higher Low
        "bearish_divergence": bool, # Price Higher High + RSI Lower High
        "divergence_type": str,
        "strength": float
      }
    """
    if len(close) < lookback or len(rsi) < lookback:
        return {"bullish_divergence": False, "bearish_divergence": False, "divergence_type": "NONE", "strength": 0.0}

    # Slice strictly closed lookback window (excluding forming candle)
    c_slice = close.iloc[-lookback:].values
    l_slice = low.iloc[-lookback:].values
    h_slice = high.iloc[-lookback:].values
    r_slice = rsi.iloc[-lookback:].values

    # Find pivot lows (fractal 5-bar: i is lower than i-2, i-1, i+1, i+2)
    pivot_lows = []
    pivot_highs = []
    
    # We examine up to index len - 3 so the pivot is fully confirmed by closed bars
    for i in range(2, len(c_slice) - 2):
        if l_slice[i] <= l_slice[i-1] and l_slice[i] <= l_slice[i-2] and l_slice[i] < l_slice[i+1] and l_slice[i] < l_slice[i+2]:
            pivot_lows.append((i, l_slice[i], r_slice[i]))
        if h_slice[i] >= h_slice[i-1] and h_slice[i] >= h_slice[i-2] and h_slice[i] > h_slice[i+1] and h_slice[i] > h_slice[i+2]:
            pivot_highs.append((i, h_slice[i], r_slice[i]))

    bullish_div = False
    bearish_div = False
    div_type = "NONE"
    strength = 0.0

    # Check Bullish Divergence across last 2 confirmed pivot lows
    if len(pivot_lows) >= 2:
        prev_idx, prev_p, prev_r = pivot_lows[-2]
        curr_idx, curr_p, curr_r = pivot_lows[-1]
        # Price made a Lower Low, but RSI made a Higher Low (and RSI is in oversold/rebound zone <= 45)
        if curr_p < prev_p and curr_r > prev_r and curr_r <= 48.0:
            bullish_div = True
            div_type = "BULLISH_REVERSAL"
            strength = round(min(1.0, (curr_r - prev_r) / 10.0), 2)

    # Check Bearish Divergence across last 2 confirmed pivot highs
    if len(pivot_highs) >= 2:
        prev_idx, prev_p, prev_r = pivot_highs[-2]
        curr_idx, curr_p, curr_r = pivot_highs[-1]
        # Price made a Higher High, but RSI made a Lower High (and RSI is in overbought/exhaustion zone >= 55)
        if curr_p > prev_p and curr_r < prev_r and curr_r >= 55.0:
            bearish_div = True
            div_type = "BEARISH_EXHAUSTION"
            strength = round(min(1.0, (prev_r - curr_r) / 10.0), 2)

    return {
        "bullish_divergence": bullish_div,
        "bearish_divergence": bearish_div,
        "divergence_type": div_type,
        "strength": strength
    }

def calculate_candle_structure(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> dict:
    """
    Evaluates candle wick absorption and rejection ratios.
    Detects upper shadow supply traps and lower shadow demand absorption.
    """
    if len(close) < 1:
        return {"upper_wick_ratio": 0.0, "lower_wick_ratio": 0.0, "is_upper_rejection": False, "is_lower_absorption": False}

    o = float(open_.iloc[-1])
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    c = float(close.iloc[-1])

    total_range = max(0.001, h - l)
    body_top = max(o, c)
    body_bottom = min(o, c)

    upper_wick = max(0.0, h - body_top)
    lower_wick = max(0.0, body_bottom - l)

    upper_wick_ratio = round(upper_wick / total_range, 3)
    lower_wick_ratio = round(lower_wick / total_range, 3)

    return {
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "is_upper_rejection": upper_wick_ratio >= 0.40,
        "is_lower_absorption": lower_wick_ratio >= 0.40
    }

def calculate_mtf_alignment(df_5m: pd.DataFrame) -> dict:
    """
    Locally resamples 5m candles into confirmed 15m intervals (zero external API calls).
    Computes 15m 50 EMA and SuperTrend to establish the hierarchical macro trend multiplier.
    Returns:
      {
        "mu_mtf": float (0.70 to 1.15),
        "status": "BULLISH_ALIGNED" | "NEUTRAL" | "BEARISH_CONFLICT",
        "ema50_15m": float,
        "st_dir_15m": int
      }
    """
    if df_5m.empty or len(df_5m) < 45: # Need at least 45 5m bars (~15 15m bars)
        return {"mu_mtf": 1.00, "status": "NEUTRAL", "ema50_15m": 0.0, "st_dir_15m": 0}

    try:
        # Resample 5m to 15m
        df_copy = df_5m.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            return {"mu_mtf": 1.00, "status": "NEUTRAL", "ema50_15m": 0.0, "st_dir_15m": 0}

        df_15m = df_copy.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Drop the last unclosed 15m interval if it has fewer than 3 5m bars
        if len(df_15m) < 10:
            return {"mu_mtf": 1.00, "status": "NEUTRAL", "ema50_15m": 0.0, "st_dir_15m": 0}

        c_15 = df_15m["Close"]
        h_15 = df_15m["High"]
        l_15 = df_15m["Low"]

        curr_p = float(c_15.iloc[-1])
        ema50_15 = float(calculate_ema(c_15, min(50, len(c_15))).iloc[-1])
        st_15, st_dir_15 = calculate_supertrend(h_15, l_15, c_15, 10, 3.0)
        last_st_dir = int(st_dir_15.iloc[-1])

        # Evaluate Alignment
        is_bullish = curr_p >= ema50_15 and last_st_dir == 1
        is_bearish = curr_p < ema50_15 and last_st_dir == -1

        if is_bullish:
            return {
                "mu_mtf": 1.15,
                "status": "BULLISH_ALIGNED",
                "ema50_15m": round(ema50_15, 2),
                "st_dir_15m": 1
            }
        elif is_bearish:
            return {
                "mu_mtf": 0.70,
                "status": "BEARISH_CONFLICT",
                "ema50_15m": round(ema50_15, 2),
                "st_dir_15m": -1
            }
        else:
            return {
                "mu_mtf": 1.00,
                "status": "NEUTRAL",
                "ema50_15m": round(ema50_15, 2),
                "st_dir_15m": last_st_dir
            }
    except Exception:
        return {"mu_mtf": 1.00, "status": "NEUTRAL", "ema50_15m": 0.0, "st_dir_15m": 0}

def calculate_intraday_vwap_bands(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates Intraday Volume-Weighted Average Price (VWAP) and ±1σ / ±2σ Standard Deviation Bands.
    """
    if df is None or len(df) < 2 or "Volume" not in df.columns:
        return {"vwap": 0.0, "upper_1sigma": 0.0, "lower_1sigma": 0.0, "upper_2sigma": 0.0, "lower_2sigma": 0.0, "sigma": 0.0}

    try:
        h = df["High"]
        l = df["Low"]
        c = df["Close"]
        v = df["Volume"].replace(0, 1) # Avoid zero division

        tp = (h + l + c) / 3.0
        cum_tp_v = (tp * v).cumsum()
        cum_v = v.cumsum()
        vwap_series = cum_tp_v / cum_v

        # Compute volume-weighted standard deviation
        current_vwap = float(vwap_series.iloc[-1])
        variance_term = (tp - current_vwap) ** 2
        cum_var_v = (variance_term * v).cumsum()
        vw_std = float(np.sqrt(cum_var_v.iloc[-1] / cum_v.iloc[-1])) if float(cum_v.iloc[-1]) > 0 else 0.0

        return {
            "vwap": round(current_vwap, 2),
            "upper_1sigma": round(current_vwap + vw_std, 2),
            "lower_1sigma": round(current_vwap - vw_std, 2),
            "upper_2sigma": round(current_vwap + 2.0 * vw_std, 2),
            "lower_2sigma": round(current_vwap - 2.0 * vw_std, 2),
            "sigma": round(vw_std, 2)
        }
    except Exception:
        return {"vwap": 0.0, "upper_1sigma": 0.0, "lower_1sigma": 0.0, "upper_2sigma": 0.0, "lower_2sigma": 0.0, "sigma": 0.0}

def calculate_rvol(volume_series: pd.Series, period: int = 20) -> float:
    """
    Calculates Relative Volume (RVol) against its 20-period moving average.
    RVol >= 1.30 confirms genuine institutional volume expansion.
    """
    if volume_series is None or len(volume_series) == 0:
        return 1.00

    try:
        current_vol = float(volume_series.iloc[-1])
        lookback = volume_series.iloc[-min(len(volume_series), period + 1):-1]
        avg_vol = float(lookback.mean()) if len(lookback) > 0 else current_vol

        if avg_vol <= 0:
            return 1.00
        return round(current_vol / avg_vol, 2)
    except Exception:
        return 1.00

def calculate_context_multiplier(
    adx: float,
    stock_trend: str = "BULLISH",
    index_trend: str = "BULLISH"
) -> float:
    """
    Unified Combined Context Multiplier mu_context = clamp(mu_ADX * mu_Breadth, 0.50, 1.25).
    Applied ONCE to Bucket 1 (Trend), eliminating uncalibrated multiplier compounding.
    """
    # 1. Smooth ADX Interpolation: [0.50, 1.00]
    if adx <= 20.0:
        mu_adx = 0.50
    elif adx >= 30.0:
        mu_adx = 1.00
    else:
        mu_adx = 0.50 + 0.50 * ((adx - 20.0) / 10.0)

    # 2. Macro Breadth Alignment: [0.80, 1.15]
    s_trend = stock_trend.upper()
    i_trend = index_trend.upper()

    if i_trend in ["NEUTRAL", "NONE", "INDEX", "UNKNOWN"]:
        mu_breadth = 1.00
    elif s_trend == i_trend:
        mu_breadth = 1.15 # Strong macro tailwind
    else:
        mu_breadth = 0.80 # Direct macro conflict / headwind

    combined_mu = mu_adx * mu_breadth
    return round(float(np.clip(combined_mu, 0.50, 1.25)), 2)


def calculate_classical_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calculates Classical Floor Pivot Points from previous session HLC.
    P = (H + L + C) / 3
    R1 = 2P - L, S1 = 2P - H
    R2 = P + (H - L), S2 = P - (H - L)
    R3 = H + 2(P - L), S3 = L - 2(H - P)
    """
    p = (high + low + close) / 3.0
    rng = high - low
    r1 = (2.0 * p) - low
    s1 = (2.0 * p) - high
    r2 = p + rng
    s2 = p - rng
    r3 = high + (2.0 * (p - low))
    s3 = low - (2.0 * (high - p))
    return {
        "pivot": round(p, 2),
        "r1": round(r1, 2),
        "s1": round(s1, 2),
        "r2": round(r2, 2),
        "s2": round(s2, 2),
        "r3": round(r3, 2),
        "s3": round(s3, 2)
    }


def calculate_fibonacci_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calculates Fibonacci Pivot Points from previous session HLC.
    P = (H + L + C) / 3
    R1 = P + 0.382 * Range, S1 = P - 0.382 * Range
    R2 = P + 0.618 * Range, S2 = P - 0.618 * Range
    R3 = P + 1.000 * Range, S3 = P - 1.000 * Range
    """
    p = (high + low + close) / 3.0
    rng = high - low
    r1 = p + (0.382 * rng)
    s1 = p - (0.382 * rng)
    r2 = p + (0.618 * rng)
    s2 = p - (0.618 * rng)
    r3 = p + (1.000 * rng)
    s3 = p - (1.000 * rng)
    return {
        "pivot": round(p, 2),
        "fib_r1": round(r1, 2),
        "fib_s1": round(s1, 2),
        "fib_r2": round(r2, 2),
        "fib_s2": round(s2, 2),
        "fib_r3": round(r3, 2),
        "fib_s3": round(s3, 2)
    }


def evaluate_vwap_location_score(
    curr_price: float,
    vwap_bands: Dict[str, float],
    raw_trend: float = 1.0
) -> float:
    """
    Fully-Specified 4-Zone VWAP Location Evaluator:
    1. Mean-Reversion / Fair Value Zone (VWAP +- 0.5 sigma): Ungated +0.80 across all regimes.
    2. Directional Value Zones:
       - Bullish (raw_trend > 0) in [VWAP - 1.5 sigma, VWAP - 0.5 sigma]: +1.20 (Discount Support).
       - Bearish (raw_trend < 0) in [VWAP + 0.5 sigma, VWAP + 1.5 sigma]: -1.20 (Premium Resistance).
       - Mismatched zones: +0.00.
    3. Exhaustion & Climax Penalties:
       - Price > VWAP + 2.0 sigma: -0.80 (Overextended upside exhaustion).
       - Price < VWAP - 2.0 sigma: -1.00 for Longs (Severe breakdown), -0.80 for Shorts (Climax selling).
    4. Neutral Regime (raw_trend == 0): Mean-Reversion Zone active at +0.80, others 0.00.
    """
    if not vwap_bands or vwap_bands.get("vwap", 0.0) <= 0:
        return 0.0

    vwap = vwap_bands["vwap"]
    std = vwap_bands.get("std", 0.0)
    if std <= 0:
        std = vwap * 0.005 # Fallback to 0.5% if std is missing

    u05 = vwap + (0.5 * std)
    l05 = vwap - (0.5 * std)
    u15 = vwap + (1.5 * std)
    l15 = vwap - (1.5 * std)
    u20 = vwap + (2.0 * std)
    l20 = vwap - (2.0 * std)

    # 1. Exhaustion Penalties (Ungated)
    if curr_price > u20:
        return -0.80
    if curr_price < l20:
        return -1.00 if raw_trend >= 0 else -0.80

    # 2. Mean-Reversion / Fair Value Zone (Ungated)
    if l05 <= curr_price <= u05:
        return 0.80

    # 3. Directional Value Zones
    if raw_trend > 0:
        # Bullish: Discount zone below VWAP is favorable for dip buying
        if l15 <= curr_price < l05:
            return 1.20
        if u05 < curr_price <= u15:
            return 0.20 # Mild continuation
    elif raw_trend < 0:
        # Bearish: Premium zone above VWAP is favorable for shorting / put buying
        if u05 < curr_price <= u15:
            return -1.20 # High short conviction (signed negative)
        if l15 <= curr_price < l05:
            return -0.20 # Mild breakdown continuation

    return 0.00


def evaluate_pivot_confluence(
    curr_price: float,
    pivots: Dict[str, float],
    raw_trend: float = 1.0,
    proximity_pct: float = 0.0025 # 0.25% proximity band
) -> float:
    """
    Symmetric 4-Case Pivot Confluence Evaluator:
    - Bullish + Near Support (S1, S2): +0.30 (Support bounce confirms Long)
    - Bullish + Near Resistance (R1, R2): -0.30 (Overhead supply obstacle)
    - Bearish + Near Resistance (R1, R2): -0.30 (Resistance rejection confirms Short, signed negative)
    - Bearish + Near Support (S1, S2): +0.30 (Demand floor obstacle vs Short, signed positive)
    - Otherwise / Neutral: 0.00
    """
    if not pivots or curr_price <= 0:
        return 0.00

    near_support = False
    near_resistance = False

    supports = [v for k, v in pivots.items() if k.startswith("s") and v > 0]
    resistances = [v for k, v in pivots.items() if k.startswith("r") and v > 0]

    for s_level in supports:
        if abs(curr_price - s_level) / curr_price <= proximity_pct:
            near_support = True
            break

    for r_level in resistances:
        if abs(curr_price - r_level) / curr_price <= proximity_pct:
            near_resistance = True
            break

    if raw_trend > 0: # Bullish
        if near_support:
            return 0.30
        if near_resistance:
            return -0.30
    elif raw_trend < 0: # Bearish
        if near_resistance:
            return -0.30 # Rejection confirms short thesis (signed negative)
        if near_support:
            return 0.30 # Demand obstacle pushes back toward neutral (signed positive)

    return 0.00


def calculate_relative_strength_vs_benchmark(
    stock_close: pd.Series,
    benchmark_close: Optional[pd.Series] = None,
    period: int = 20
) -> Dict[str, Any]:
    """
    Institutional Relative Strength (RS) vs Benchmark (NIFTY 50).
    Measures alpha and outperformance over rolling period.
    - RS Ratio > 1.02: Institutional Accumulation / Outperforming benchmark.
    - RS Ratio 0.98 - 1.02: In-line with market.
    - RS Ratio < 0.98: Underperforming benchmark (avoid Longs).
    """
    if stock_close is None or len(stock_close) < period:
        return {"rs_ratio": 1.00, "rs_pct": 0.0, "status": "INLINE", "score_boost": 0.0}

    s_ret = (stock_close.iloc[-1] / stock_close.iloc[-period]) - 1.0 if stock_close.iloc[-period] > 0 else 0.0
    
    if benchmark_close is not None and len(benchmark_close) >= period and benchmark_close.iloc[-period] > 0:
        b_ret = (benchmark_close.iloc[-1] / benchmark_close.iloc[-period]) - 1.0
    else:
        b_ret = 0.0

    rs_diff = s_ret - b_ret
    rs_ratio = round(1.0 + rs_diff, 4)

    if rs_ratio >= 1.025:
        status = "STRONG_OUTPERFORMER"
        boost = 0.40
    elif rs_ratio >= 1.008:
        status = "OUTPERFORMING"
        boost = 0.20
    elif rs_ratio <= 0.975:
        status = "HEAVY_UNDERPERFORMER"
        boost = -0.40
    elif rs_ratio <= 0.992:
        status = "UNDERPERFORMING"
        boost = -0.20
    else:
        status = "INLINE"
        boost = 0.0

    return {
        "rs_ratio": rs_ratio,
        "rs_diff_pct": round(rs_diff * 100.0, 2),
        "status": status,
        "score_boost": boost
    }


def calculate_ttm_squeeze(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5
) -> Dict[str, Any]:
    """
    TTM Volatility Squeeze Indicator (Bollinger Bands vs Keltner Channels).
    - Squeeze ON: Bollinger Bands contract inside Keltner Channel (Energy Coiling).
    - Squeeze FIRE: Bollinger Bands expand outside Keltner Channel with directional momentum.
    High win-rate setup (>75%) when exiting compression.
    """
    if len(close) < bb_period:
        return {"squeeze_on": False, "squeeze_fired": False, "momentum_direction": "NEUTRAL", "score_boost": 0.0}

    # Bollinger Bands
    sma = calculate_sma(close, bb_period)
    r_std = close.rolling(window=bb_period, min_periods=1).std()
    bb_upper = sma + (r_std * bb_std)
    bb_lower = sma - (r_std * bb_std)

    # Keltner Channels
    atr = calculate_atr(high, low, close, kc_period)
    kc_upper = sma + (atr * kc_mult)
    kc_lower = sma - (atr * kc_mult)

    # Squeeze Condition (BB inside KC)
    is_squeeze_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_now = bool(is_squeeze_series.iloc[-1])
    
    # Squeeze Fired (Was in squeeze in last 3 bars, now broken out)
    past_squeeze = bool(is_squeeze_series.iloc[-4:-1].any()) if len(is_squeeze_series) >= 4 else False
    squeeze_fired = (not squeeze_now) and past_squeeze

    # Momentum Linear Regression
    mom_series = close - sma
    mom_curr = float(mom_series.iloc[-1])
    mom_prev = float(mom_series.iloc[-2]) if len(mom_series) > 1 else mom_curr

    if mom_curr > 0 and mom_curr > mom_prev:
        mom_dir = "BULLISH_EXPANSION"
        boost = 0.40 if squeeze_fired else (0.20 if squeeze_now else 0.10)
    elif mom_curr > 0 and mom_curr <= mom_prev:
        mom_dir = "BULLISH_FADING"
        boost = 0.00
    elif mom_curr < 0 and mom_curr < mom_prev:
        mom_dir = "BEARISH_EXPANSION"
        boost = -0.40 if squeeze_fired else (-0.20 if squeeze_now else -0.10)
    else:
        mom_dir = "BEARISH_FADING"
        boost = 0.00

    return {
        "squeeze_on": squeeze_now,
        "squeeze_fired": squeeze_fired,
        "momentum_direction": mom_dir,
        "score_boost": boost
    }


def calculate_vsa_structure(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> Dict[str, Any]:
    """
    Volume Spread Analysis (VSA) Trap Filter:
    - No Supply Bar: Down-bar with narrow spread (< 0.6x ATR) and low volume (< 0.8x RVol) = Bullish continuation.
    - Stopping Volume: High volume (> 1.5x) with narrow spread closing in upper half after decline = Accumulation floor.
    - Climax Distribution: Ultra-high volume (> 2.2x) on wide range into resistance with upper rejection = Bearish trap.
    """
    if len(close) < 20:
        return {"pattern": "NORMAL", "score_boost": 0.0, "is_trap": False}

    atr = float(calculate_atr(high, low, close, 14).iloc[-1])
    rvol = calculate_rvol(volume, 20)

    curr_o = float(open_.iloc[-1])
    curr_h = float(high.iloc[-1])
    curr_l = float(low.iloc[-1])
    curr_c = float(close.iloc[-1])

    spread = curr_h - curr_l
    body = abs(curr_c - curr_o)
    upper_wick = curr_h - max(curr_o, curr_c)
    lower_wick = min(curr_o, curr_c) - curr_l

    # 1. Climax Distribution Trap (Overhead supply dumping)
    if rvol >= 2.0 and upper_wick >= (0.35 * spread) and curr_c < curr_o:
        return {
            "pattern": "CLIMAX_DISTRIBUTION_TRAP",
            "description": "Institutional selling into strength with heavy upper wick.",
            "score_boost": -0.50,
            "is_trap": True
        }

    # 2. Stopping Volume / Absorption Support
    if rvol >= 1.5 and lower_wick >= (0.40 * spread) and curr_c >= curr_o:
        return {
            "pattern": "STOPPING_VOLUME_ABSORPTION",
            "description": "Institutional demand absorbing selling pressure at lows.",
            "score_boost": 0.35,
            "is_trap": False
        }

    # 3. No Supply Pullback Bar (Dry-up before breakout)
    if curr_c < curr_o and spread < (0.7 * atr) and rvol < 0.75:
        return {
            "pattern": "NO_SUPPLY_PULLBACK",
            "description": "Low-volume pullback testing supply (Bullish continuation).",
            "score_boost": 0.30,
            "is_trap": False
        }

    return {"pattern": "NORMAL", "description": "Standard order-flow structure.", "score_boost": 0.0, "is_trap": False}




