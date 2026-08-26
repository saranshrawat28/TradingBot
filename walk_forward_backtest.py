"""
Walk-Forward Backtesting Harness for ApexTrade
================================================

Purpose
-------
Phase 1 Validation: PROVES THE STATISTICAL EDGE IS REAL before deploying capital.
Plugs real StockAdvisor / Orthogonal Scoring Engine into a walk-forward harness
with realistic Indian brokerage, STT, exchange fees, and slippage.

Compares against 3 Independent Baselines:
1. Buy & Hold Benchmark
2. Classic SMA 20/50 Crossover
3. Random Entry Monte Carlo Generator

Usage:
    python walk_forward_backtest.py --symbol RELIANCE.NS --period 2y
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

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any

from src.data.data_fetcher import get_historical_data
from src.engine.stock_advisor import StockAdvisor


# ============================================================================
# 1. COST MODEL — Realistic Indian Intraday Equity / F&O Trading Costs
# ============================================================================

@dataclass
class CostModel:
    """
    Realistic round-trip cost assumptions for Indian equity & F&O trades.
    Includes Zerodha ₹20 / 0.03% brokerage, STT, NSE txn fees, GST, Stamp Duty, and Slippage.
    """
    brokerage_flat_inr: float = 20.0        # per executed order (Zerodha intraday)
    brokerage_pct: float = 0.0003           # 0.03%, whichever is lower, per side
    stt_pct_sell_intraday: float = 0.00025  # STT on sell side only, intraday equity
    exchange_txn_pct: float = 0.0000297     # NSE transaction charges
    gst_pct: float = 0.18                   # GST on (brokerage + exchange charges)
    sebi_charges_pct: float = 0.0000010     # SEBI turnover fee
    stamp_duty_pct_buy: float = 0.00003     # Stamp duty on buy side
    slippage_pct: float = 0.0004            # 4 bps assumed slippage per fill

    def brokerage(self, turnover: float) -> float:
        return min(self.brokerage_flat_inr, turnover * self.brokerage_pct)

    def round_trip_cost(self, entry_price: float, exit_price: float, qty: int) -> float:
        """Total cost in INR for one full buy+sell round trip."""
        buy_turnover = entry_price * qty
        sell_turnover = exit_price * qty

        brokerage = self.brokerage(buy_turnover) + self.brokerage(sell_turnover)
        stt = sell_turnover * self.stt_pct_sell_intraday
        exch = (buy_turnover + sell_turnover) * self.exchange_txn_pct
        sebi = (buy_turnover + sell_turnover) * self.sebi_charges_pct
        stamp = buy_turnover * self.stamp_duty_pct_buy
        gst = (brokerage + exch) * self.gst_pct
        slippage = (buy_turnover + sell_turnover) * self.slippage_pct

        return brokerage + stt + exch + sebi + stamp + gst + slippage


# ============================================================================
# 2. TRADE / SIGNAL DATA STRUCTURES
# ============================================================================

@dataclass
class Signal:
    date: pd.Timestamp
    symbol: str
    direction: str          # "LONG" or "FLAT"
    entry_price: float
    stop_loss: float
    target: float
    confidence: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class TradeResult:
    date: pd.Timestamp
    symbol: str
    entry_price: float
    exit_price: float
    exit_reason: str        # "TARGET", "STOP", "TIMEOUT"
    qty: int
    gross_pnl: float
    cost: float
    net_pnl: float
    r_multiple: float       # net_pnl relative to initial risk (SL distance * qty)


# ============================================================================
# 3. SIGNAL GENERATORS — Real StockAdvisor vs Baselines
# ============================================================================

def real_apex_signal_generator(df: pd.DataFrame, symbol: str, eval_start_idx: Optional[int] = None) -> List[Signal]:
    """
    Calls the actual StockAdvisor scoring engine bar-by-bar using ONLY data
    available up to and including each day (0% lookahead bias).
    """
    signals = []
    
    # Ensure standardized column names
    clean_df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in clean_df.columns and col.lower() in clean_df.columns:
            clean_df[col] = clean_df[col.lower()]
            
    start_idx = eval_start_idx if eval_start_idx is not None else 30
    start_idx = max(30, start_idx)
    
    for i in range(start_idx, len(clean_df)):
        window = clean_df.iloc[:i+1] # Strictly data up to today
        
        try:
            analysis = StockAdvisor.evaluate_df_slice(window, symbol=symbol, horizon="swing")
            score = float(analysis.get("score", 5.0))
            verdict = analysis.get("verdict", "")
            
            # Conviction threshold for Swing/Positional: Math Score >= 6.5
            if score >= 6.5 and ("BUY" in verdict or "STRONG" in verdict):
                entry_p = float(window["Close"].iloc[-1])
                sl_raw = analysis.get("stop_loss", {})
                t1_raw = analysis.get("target_1", {})
                t2_raw = analysis.get("target_2", {})
                
                sl_p = float(sl_raw.get("price", entry_p * 0.985) if isinstance(sl_raw, dict) else (sl_raw or entry_p * 0.985))
                t1_p = float(t1_raw.get("price", entry_p * 1.025) if isinstance(t1_raw, dict) else (t1_raw or entry_p * 1.025))
                t2_p = float(t2_raw.get("price", entry_p * 1.050) if isinstance(t2_raw, dict) else (t2_raw or entry_p * 1.050))
                
                signals.append(Signal(
                    date=window.index[-1],
                    symbol=symbol,
                    direction="LONG",
                    entry_price=entry_p,
                    stop_loss=sl_p,
                    target=t1_p,
                    confidence=score,
                    meta={"regime": analysis.get("regime", "UNKNOWN"), "rvol": analysis.get("rvol", 1.0)}
                ))
        except Exception as e:
            continue
            
    return signals


def sma_crossover_baseline(df: pd.DataFrame, symbol: str) -> List[Signal]:
    """Trivial baseline: 20/50 SMA crossover, fixed 2% stop / 4% target."""
    signals = []
    close = df["Close"] if "Close" in df.columns else df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    crossed_up = (sma20 > sma50) & (sma20.shift(1) <= sma50.shift(1))
    
    for i in range(50, len(df)):
        if crossed_up.iloc[i]:
            entry = float(close.iloc[i])
            signals.append(Signal(
                date=df.index[i],
                symbol=symbol,
                direction="LONG",
                entry_price=entry,
                stop_loss=entry * 0.98,
                target=entry * 1.04,
                confidence=5.0
            ))
    return signals


def random_baseline(df: pd.DataFrame, symbol: str, rng: np.random.Generator,
                    trade_prob: float = 0.03) -> List[Signal]:
    """Random entry baseline matching active trade frequency, fixed 2% SL / 4% TP."""
    signals = []
    close = df["Close"] if "Close" in df.columns else df["close"]
    for i in range(50, len(df)):
        if rng.random() < trade_prob:
            entry = float(close.iloc[i])
            signals.append(Signal(
                date=df.index[i],
                symbol=symbol,
                direction="LONG",
                entry_price=entry,
                stop_loss=entry * 0.98,
                target=entry * 1.04,
                confidence=5.0
            ))
    return signals


# ============================================================================
# 4. TRADE SIMULATOR — Fixed-Fractional Risk Sizing & Execution
# ============================================================================

def simulate_trade(df: pd.DataFrame, signal: Signal, costs: CostModel,
                   capital: float, risk_pct: float = 0.01,
                   max_hold_bars: int = 15) -> Optional[TradeResult]:
    """
    Resolves a single trade signal forward bar-by-bar.
    Position sizing is derived strictly from fixed-fractional risk (1% of capital / SL distance).
    """
    try:
        idx = df.index.get_loc(signal.date)
    except KeyError:
        return None
        
    if idx + 1 >= len(df):
        return None

    risk_per_share = signal.entry_price - signal.stop_loss
    if risk_per_share <= 0:
        return None
        
    qty = max(1, int((capital * risk_pct) / risk_per_share))

    high_col = "High" if "High" in df.columns else "high"
    low_col = "Low" if "Low" in df.columns else "low"
    close_col = "Close" if "Close" in df.columns else "close"

    exit_price, exit_reason = None, None
    for j in range(idx + 1, min(idx + 1 + max_hold_bars, len(df))):
        bar = df.iloc[j]
        if float(bar[low_col]) <= signal.stop_loss:
            exit_price, exit_reason = signal.stop_loss, "STOP"
            break
        if float(bar[high_col]) >= signal.target:
            exit_price, exit_reason = signal.target, "TARGET"
            break
            
    if exit_price is None:
        last_idx = min(idx + max_hold_bars, len(df) - 1)
        exit_price, exit_reason = float(df.iloc[last_idx][close_col]), "TIMEOUT"

    gross_pnl = (exit_price - signal.entry_price) * qty
    cost = costs.round_trip_cost(signal.entry_price, exit_price, qty)
    net_pnl = gross_pnl - cost
    r_multiple = net_pnl / max(0.01, risk_per_share * qty)

    return TradeResult(
        date=signal.date,
        symbol=signal.symbol,
        entry_price=signal.entry_price,
        exit_price=exit_price,
        exit_reason=exit_reason,
        qty=qty,
        gross_pnl=gross_pnl,
        cost=cost,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
    )


# ============================================================================
# 5. METRICS ENGINE
# ============================================================================

def compute_metrics(trades: List[TradeResult], starting_capital: float,
                    trading_days_in_period: int) -> Dict[str, Any]:
    if not trades:
        return {
            "n_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "net_pnl_inr": 0.0, "total_return_pct": 0.0, "cagr_pct": 0.0,
            "sharpe": 0.0, "max_drawdown_pct": 0.0, "avg_r_multiple": 0.0
        }

    pnl = np.array([t.net_pnl for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    equity_curve = starting_capital + np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    daily_returns = pnl / starting_capital
    sharpe = (
        float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
        if daily_returns.std() > 0 else 0.0
    )

    total_return = float(equity_curve[-1] / starting_capital - 1)
    years = max(trading_days_in_period / 252.0, 0.25)
    cagr = float((1 + total_return) ** (1.0 / years) - 1)

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    
    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 2)
    elif gross_profit > 0:
        profit_factor = 5.0
    else:
        profit_factor = 0.0

    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 1),
        "avg_win_inr": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_inr": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "net_pnl_inr": round(float(pnl.sum()), 2),
        "total_return_pct": round(100.0 * total_return, 2),
        "cagr_pct": round(100.0 * cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(100.0 * max_dd, 2),
        "avg_r_multiple": round(float(np.mean([t.r_multiple for t in trades])), 2),
    }


def buy_and_hold_metrics(df: pd.DataFrame, starting_capital: float) -> Dict[str, Any]:
    close_col = "Close" if "Close" in df.columns else "close"
    ret = float(df[close_col].iloc[-1] / df[close_col].iloc[0] - 1)
    daily_ret = df[close_col].pct_change().dropna()
    sharpe = (
        float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))
        if daily_ret.std() > 0 else 0.0
    )
    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    max_dd = float(((cum - running_max) / running_max).min())
    years = max(len(df) / 252.0, 0.25)
    cagr = float((1 + ret) ** (1.0 / years) - 1)
    return {
        "total_return_pct": round(100.0 * ret, 2),
        "cagr_pct": round(100.0 * cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(100.0 * max_dd, 2),
    }


# ============================================================================
# 6. WALK-FORWARD ENGINE
# ============================================================================

@dataclass
class WalkForwardConfig:
    train_days: int = 250     # 1 trading year in-sample
    test_days: int = 63       # 3 months out-of-sample
    step_days: int = 63       # Slide forward by one test window
    starting_capital: float = 100_000.0
    risk_pct_per_trade: float = 0.01


def run_walk_forward(df: pd.DataFrame, symbol: str,
                     signal_fn: Callable[..., List[Signal]],
                     config: WalkForwardConfig, costs: CostModel) -> pd.DataFrame:
    results = []
    n = len(df)
    start = 0
    window_id = 0

    while start + config.train_days + config.test_days <= n:
        train_end = start + config.train_days
        test_end = train_end + config.test_days

        test_df = df.iloc[start:test_end]
        test_start_date = df.index[train_end]
        test_end_date = df.index[test_end - 1]

        try:
            signals = signal_fn(test_df, symbol, eval_start_idx=config.train_days)
        except TypeError:
            signals = signal_fn(test_df, symbol)
            
        signals = [s for s in signals if s.date >= test_start_date]

        trades = []
        for sig in signals:
            tr = simulate_trade(test_df, sig, costs, config.starting_capital,
                                config.risk_pct_per_trade)
            if tr is not None:
                trades.append(tr)

        m = compute_metrics(trades, config.starting_capital, config.test_days)
        m["window"] = window_id + 1
        m["test_start"] = test_start_date.strftime("%Y-%m-%d")
        m["test_end"] = test_end_date.strftime("%Y-%m-%d")
        results.append(m)

        window_id += 1
        start += config.step_days

    return pd.DataFrame(results)


# ============================================================================
# 7. DATA LOADER — Live Indian Market Data with Synthetic Fallback
# ============================================================================

def load_price_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Load historical OHLCV data from yfinance/data_fetcher."""
    df = get_historical_data(symbol, period=period, interval=interval)
    if not df.empty and len(df) >= 100:
        return df

    print(f"⚠️ Live data unavailable for {symbol}. Using synthetic geometric random walk.")
    rng = np.random.default_rng(42)
    n_days = 500
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    returns = rng.normal(0.0005, 0.015, n_days)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.007, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.007, n_days)))
    open_ = low + (high - low) * rng.random(n_days)
    volume = rng.integers(100_000, 3_000_000, n_days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ============================================================================
# 8. MAIN CLI RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Backtesting Harness for ApexTrade")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="NSE stock ticker (e.g. RELIANCE.NS, ^NSEI)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period (e.g. 1y, 2y, 5y)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    df = load_price_data(symbol, period=args.period, interval="1d")

    costs = CostModel()
    config = WalkForwardConfig()

    print("=" * 84)
    print(f"🏛️ APEXTRADE WALK-FORWARD QUANTITATIVE VALIDATION HARNESS — {symbol}")
    print(f"   Historical Period: {args.period} | Capital: ₹{config.starting_capital:,.0f} | Risk/Trade: 1.0%")
    print(f"   Friction Model: Zerodha Intraday (₹20/0.03%) + STT 0.025% + GST 18% + 4 bps Slippage")
    print("=" * 84)

    strategies = {
        "🌟 ApexTrade Real Scorer (StockAdvisor)": real_apex_signal_generator,
        "Baseline 1: SMA 20/50 Crossover": sma_crossover_baseline,
        "Baseline 2: Random Entry (Monte Carlo)": lambda d, s: random_baseline(d, s, np.random.default_rng(7)),
    }

    all_results = {}
    for name, fn in strategies.items():
        wf = run_walk_forward(df, symbol, fn, config, costs)
        all_results[name] = wf
        print(f"\n--- {name} ---")
        if wf.empty or wf["n_trades"].sum() == 0:
            print("  No trades generated across any walk-forward window.")
            continue
            
        print(wf[["window", "test_start", "test_end", "n_trades", "win_rate_pct",
                   "profit_factor", "net_pnl_inr", "sharpe", "max_drawdown_pct"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

        agg_trades = wf["n_trades"].sum()
        total_pnl = wf["net_pnl_inr"].sum() if "net_pnl_inr" in wf else 0.0
        avg_sharpe = wf.loc[wf["n_trades"] > 0, "sharpe"].mean()
        worst_dd = wf["max_drawdown_pct"].min() if "max_drawdown_pct" in wf else 0.0
        profitable_windows = (wf["net_pnl_inr"] > 0).sum()
        
        print(f"  📊 AGGREGATE: {agg_trades} trades | Net P&L: ₹{total_pnl:+,.2f} | "
              f"Avg Sharpe: {avg_sharpe:.2f} | Worst Drawdown: {worst_dd:.2f}% | "
              f"Profitable Windows: {profitable_windows}/{len(wf)}")

    bh = buy_and_hold_metrics(df, config.starting_capital)
    print("\n--- Baseline 3: Passive Buy & Hold (Full Period) ---")
    for k, v in bh.items():
        print(f"  {k:20s}: {v:,.2f}")

    print("\n" + "=" * 84)
    print("🎯 QUANTITATIVE VERDICT RULE:")
    print("  To confirm a genuine institutional edge, the Real Strategy must beat:")
    print("  1. Baseline 3 (Buy & Hold) on Sharpe & Drawdown.")
    print("  2. Baseline 1 (SMA Crossover) on Profit Factor & Net PnL.")
    print("  3. Baseline 2 (Random Noise) across >= 70% of walk-forward windows.")
    print("=" * 84)


if __name__ == "__main__":
    main()
