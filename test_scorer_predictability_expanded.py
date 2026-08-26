"""
Expanded Scorer Predictability Test — Multi-Symbol, Bootstrap-Validated
=========================================================================

Evaluates whether the StockAdvisor scoring engine has a genuine, statistically 
significant relationship with forward returns across a diversified basket of 
20 liquid NSE stocks over 5 years of daily data with 2,000-iteration bootstrap CIs.
"""

from __future__ import annotations

import sys
import os

# Enable UTF-8 for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ============================================================================
# CONFIG
# ============================================================================

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "NESTLEIND.NS", "ONGC.NS",
]

HISTORY_YEARS = 5           # 5 years of daily historical data (~1,250 bars/symbol)
FORWARD_HORIZON_BARS = 10   # 10 daily bars ≈ 2 weeks forward horizon
STEP_BARS = 5               # Stride by 1 week to reduce autocorrelation & optimize runtime
MIN_LOOKBACK_BARS = 210     # Enough lookback bars for 200 EMA and pivots
N_BOOTSTRAP = 2000          # Resamples for 90% confidence intervals

SCORE_BANDS = [
    (-999, 5.0, "< 5.0 (Bearish)"),
    (5.0, 6.0, "5.0-5.9 (Neutral)"),
    (6.0, 7.0, "6.0-6.9 (Mild Trend)"),
    (7.0, 7.5, "7.0-7.4 (Sweet Spot?)"),
    (7.5, 8.0, "7.5-7.9 (High)"),
    (8.0, 999, ">= 8.0 (Elite/Exhaustion?)"),
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_price_data(symbol: str, years: int) -> Optional[pd.DataFrame]:
    """
    Loads daily historical OHLCV data using project data fetcher with yfinance fallback.
    Standardizes on capitalized column names expected by StockAdvisor.
    """
    try:
        from src.data.data_fetcher import get_historical_data
        df = get_historical_data(symbol, period=f"{years}y", interval="1d")
        if df is not None and not df.empty and len(df) >= MIN_LOOKBACK_BARS + FORWARD_HORIZON_BARS:
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns and col.lower() in df.columns:
                    df[col] = df[col.lower()]
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        pass

    try:
        import yfinance as yf
        df = yf.download(symbol, period=f"{years}y", interval="1d", progress=False)
        if df is not None and not df.empty:
            clean_cols = {}
            for c in df.columns:
                name = c[0] if isinstance(c, tuple) else c
                clean_cols[c] = name.capitalize()
            df = df.rename(columns=clean_cols)
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        pass

    return None


# ============================================================================
# SCORER HOOK
# ============================================================================

def get_score_for_window(window: pd.DataFrame, symbol: str) -> Optional[float]:
    """
    Calls the real StockAdvisor scoring engine bar-by-bar (0% look-ahead).
    Correctly parses Python dict return type.
    """
    try:
        from src.engine.stock_advisor import StockAdvisor
        res = StockAdvisor.evaluate_df_slice(window, symbol=symbol, horizon="swing")
        if isinstance(res, dict):
            return float(res.get("math_score") or res.get("score", 5.0))
        return float(getattr(res, "math_score", None) or getattr(res, "score", 5.0))
    except Exception:
        return None


# ============================================================================
# SAMPLING & BOOTSTRAP ENGINE
# ============================================================================

@dataclass
class Sample:
    symbol: str
    date: pd.Timestamp
    score: float
    forward_return: float   # % return over FORWARD_HORIZON_BARS
    hit: bool                # forward_return > 0


def collect_samples_for_symbol(symbol: str) -> List[Sample]:
    df = load_price_data(symbol, HISTORY_YEARS)
    if df is None or len(df) < MIN_LOOKBACK_BARS + FORWARD_HORIZON_BARS + 10:
        print(f"  [skip] {symbol}: insufficient data")
        return []

    close = df["Close"].values
    samples = []

    for i in range(MIN_LOOKBACK_BARS, len(df) - FORWARD_HORIZON_BARS, STEP_BARS):
        window = df.iloc[:i + 1]
        score = get_score_for_window(window, symbol)
        if score is None:
            continue

        entry_price = float(close[i])
        exit_price = float(close[i + FORWARD_HORIZON_BARS])
        fwd_ret = ((exit_price / max(0.01, entry_price)) - 1.0) * 100.0

        samples.append(Sample(
            symbol=symbol,
            date=df.index[i],
            score=score,
            forward_return=fwd_ret,
            hit=fwd_ret > 0.0,
        ))

    return samples


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = N_BOOTSTRAP,
                  ci: float = 0.90) -> Tuple[float, float, float]:
    """Returns (point_estimate, lower_bound, upper_bound) for a statistic."""
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat_fn(values))
    boots = np.array([
        float(stat_fn(np.random.choice(values, size=len(values), replace=True)))
        for _ in range(n_boot)
    ])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return (point, point, point)
    lo = float(np.percentile(boots, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(boots, (1.0 + ci) / 2.0 * 100.0))
    return point, lo, hi


def summarize_by_band(samples: List[Sample]) -> pd.DataFrame:
    rows = []
    for lo, hi, label in SCORE_BANDS:
        band_samples = [s for s in samples if lo <= s.score < hi]
        n = len(band_samples)
        if n == 0:
            rows.append({
                "band": label, "n": 0, "n_symbols": 0, "win_rate": None,
                "win_rate_ci": None, "avg_return": None, "avg_return_ci": None
            })
            continue

        returns = np.array([s.forward_return for s in band_samples])
        hits = np.array([1.0 if s.hit else 0.0 for s in band_samples])
        n_symbols = len(set(s.symbol for s in band_samples))

        wr_point, wr_lo, wr_hi = bootstrap_ci(hits, np.mean)
        ret_point, ret_lo, ret_hi = bootstrap_ci(returns, np.mean)

        rows.append({
            "band": label,
            "n": n,
            "n_symbols": n_symbols,
            "win_rate": f"{wr_point * 100.0:.1f}%",
            "win_rate_ci": f"[{wr_lo * 100.0:.1f}%, {wr_hi * 100.0:.1f}%]",
            "avg_return": f"{ret_point:+.2f}%",
            "avg_return_ci": f"[{ret_lo:+.2f}%, {ret_hi:+.2f}%]",
        })
    return pd.DataFrame(rows)


def overall_correlation(samples: List[Sample]) -> Tuple[float, float, float]:
    """Pearson correlation between score and forward return, with bootstrap CI."""
    if len(samples) < 10:
        return (float("nan"), float("nan"), float("nan"))
    scores = np.array([s.score for s in samples])
    returns = np.array([s.forward_return for s in samples])

    def corr_fn(idx):
        if len(set(idx)) < 2:
            return 0.0
        val = np.corrcoef(scores[idx], returns[idx])[0, 1]
        return 0.0 if np.isnan(val) else float(val)

    n = len(samples)
    raw_point = np.corrcoef(scores, returns)[0, 1]
    point = 0.0 if np.isnan(raw_point) else float(raw_point)

    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, size=n, replace=True)
        boots.append(corr_fn(idx))
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    lo = float(np.percentile(boots, 5))
    hi = float(np.percentile(boots, 95))
    return point, lo, hi


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 92)
    print(f"EXPANDED SCORER PREDICTABILITY TEST — {len(SYMBOLS)} symbols, "
          f"{HISTORY_YEARS}y history, {FORWARD_HORIZON_BARS}-bar forward horizon, {STEP_BARS}-bar step")
    print(f"Confidence Level: 90% | Bootstrap Resamples: {N_BOOTSTRAP:,}")
    print("=" * 92)

    all_samples: List[Sample] = []
    for idx, sym in enumerate(SYMBOLS, 1):
        print(f"[{idx:02d}/{len(SYMBOLS)}] Processing {sym:14s}...", end=" ", flush=True)
        samples = collect_samples_for_symbol(sym)
        print(f"-> {len(samples)} valid samples collected")
        all_samples.extend(samples)

    if not all_samples:
        print("\nNo samples collected. Check StockAdvisor import path and data loader.")
        return

    unique_syms = len(set(s.symbol for s in all_samples))
    print("\n" + "=" * 92)
    print(f"TOTAL SAMPLES COLLECTED: {len(all_samples):,} across {unique_syms} symbols")
    print("=" * 92 + "\n")

    summary = summarize_by_band(all_samples)
    print(summary.to_string(index=False))

    point, lo, hi = overall_correlation(all_samples)
    print("\n" + "-" * 92)
    print(f"Overall Pearson Correlation (Score vs Forward Return): "
          f"{point:+.3f}  [90% CI: {lo:+.3f}, {hi:+.3f}]")
    print("-" * 92)

    print("\n" + "=" * 92)
    print("EMPIRICAL VERDICT GUIDELINES:")
    print("=" * 92)
    print("""
1. Check sample sizes: If any band has n < 50 or n_symbols < 10, it is inconclusive.
2. Compare confidence intervals across bands:
   - If [wr_lo, wr_hi] for '7.0-7.4' overlaps heavily with '>= 8.0', the difference is noise.
   - If '7.0-7.4' lower bound is strictly above '>= 8.0' upper bound, exhaustion is proven.
3. Check correlation CI:
   - If 90% CI straddles zero (e.g. [-0.03, +0.04]), the score has no directional edge.
   - If lower bound > 0.00, genuine monotonic predictive power is confirmed.
""")


if __name__ == "__main__":
    main()
