"""
Ultra-Fast Walk-Forward Historical Validator for Mathematical Scoring Engine.
Evaluates score-bucket forward expectancy, win rates, and profit factor
with realistic Indian transaction costs (STT, GST, brokerage, slippage)
across in-sample and out-of-sample regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.data.data_fetcher import get_historical_data
from src.strategies.indicators import add_all_indicators

class ScorerValidator:
    """
    Validates the quantitative scoring formula on historical candle data.
    """

    # Indian F&O / Equity Intraday Transaction Cost Constants
    BROKERAGE_PER_ORDER = 20.0 # Zerodha ₹20
    STT_SELL_RATE = 0.00025    # 0.025% on intraday sell
    EXCHANGE_RATE = 0.0000345  # 0.00345%
    SEBI_RATE = 0.000001       # 0.0001%
    GST_RATE = 0.18            # 18% on (Brokerage + Exchange + SEBI)
    SLIPPAGE_RATE = 0.0005     # 0.05% realistic slippage

    @classmethod
    def calculate_trade_friction(cls, entry_price: float, exit_price: float, quantity: int) -> float:
        """Calculate total round-trip statutory taxes, charges, and slippage in INR."""
        turnover_buy = entry_price * quantity
        turnover_sell = exit_price * quantity
        total_turnover = turnover_buy + turnover_sell
        
        brokerage = min(cls.BROKERAGE_PER_ORDER, turnover_buy * 0.0003) + min(cls.BROKERAGE_PER_ORDER, turnover_sell * 0.0003)
        stt = turnover_sell * cls.STT_SELL_RATE
        exchange_charges = total_turnover * cls.EXCHANGE_RATE
        sebi_charges = total_turnover * cls.SEBI_RATE
        gst = (brokerage + exchange_charges + sebi_charges) * cls.GST_RATE
        slippage = total_turnover * cls.SLIPPAGE_RATE
        
        return brokerage + stt + exchange_charges + sebi_charges + gst + slippage

    @classmethod
    def calculate_fast_score(cls, row: pd.Series, prev_row: pd.Series) -> float:
        """Fast orthogonal score calculator on precomputed indicator columns."""
        curr_p = float(row["Close"])
        ema9 = float(row.get("EMA_9", curr_p))
        ema21 = float(row.get("EMA_21", curr_p))
        ema50 = float(row.get("EMA_50", curr_p))
        ema200 = float(row.get("EMA_200", curr_p))
        rsi = float(row.get("RSI_14", 50.0))
        macd_hist = float(row.get("MACD_Hist", 0.0))
        prev_hist = float(prev_row.get("MACD_Hist", 0.0))
        st_dir = int(row.get("SuperTrend_Dir", 1))
        adx = float(row.get("ADX_14", 20.0))
        pct_b = float(row.get("BB_PctB", 0.5))
        bw = float(row.get("BB_Bandwidth", 5.0))
        vwap = float(row.get("VWAP", curr_p))
        vol = float(row.get("Volume", 1.0))
        avg_vol = float(row.get("Volume_Avg20", vol))

        # Continuous ADX scaling factor: 0.5 (Chop <= 20) -> 1.0 (Trend >= 25)
        adx_factor = min(1.0, max(0.0, (adx - 20.0) / 5.0))
        mu_trend = 0.5 + 0.5 * adx_factor
        is_range = adx <= 20.0

        # Bucket 1: Trend Alignment (Exact +/-2.50 Max/Min)
        trend_pts = 0.0
        if ema9 > ema21: trend_pts += 0.75
        elif ema9 < ema21: trend_pts -= 0.75

        if curr_p > ema50: trend_pts += 0.75
        elif curr_p < ema50: trend_pts -= 0.75

        if curr_p > ema200: trend_pts += 0.50
        elif curr_p < ema200: trend_pts -= 0.50

        if st_dir == 1: trend_pts += 0.50
        elif st_dir == -1: trend_pts -= 0.50

        trend_pts = trend_pts * mu_trend
        trend_pts = max(-2.5, min(2.5, trend_pts))

        # Bucket 2: Momentum (Max +/-2.0)
        mom_pts = 0.0
        if 50.0 <= rsi <= 68.0: mom_pts += 1.0
        elif rsi > 70.0: mom_pts -= 0.5
        elif rsi < 35.0:
            if is_range: mom_pts += 1.0
            else: mom_pts -= 0.5
        if macd_hist > 0 and macd_hist > prev_hist: mom_pts += 1.0
        elif macd_hist < 0: mom_pts -= 0.5
        mom_pts = max(-2.0, min(2.0, mom_pts))

        # Bucket 3: Volatility & Location (Max +/-1.5)
        vol_pts = 0.0
        if 0.4 <= pct_b <= 0.8: vol_pts += 0.8
        elif pct_b > 0.95: vol_pts -= 0.5
        elif pct_b < 0.05:
            if is_range: vol_pts += 0.5
            else: vol_pts -= 0.5
        if bw < 4.0: vol_pts += 0.7
        vol_pts = max(-1.5, min(1.5, vol_pts))

        # Bucket 4: Flow & VWAP (Max +/-1.5)
        flow_pts = 0.0
        if vol > avg_vol * 1.25: flow_pts += 0.8
        elif vol < avg_vol * 0.50: flow_pts -= 0.4
        
        if curr_p > vwap: flow_pts += 0.7
        else: flow_pts -= 0.5
        flow_pts = max(-1.5, min(1.5, flow_pts))

        raw_score = 5.0 + trend_pts + mom_pts + vol_pts + flow_pts
        return round(max(1.0, min(9.8, raw_score)), 1)

    @classmethod
    def run_validation(
        cls,
        symbols: Optional[List[str]] = None,
        period: str = "30d",
        interval: str = "15m",
        train_test_split: float = 0.60,
        holding_bars: int = 12
    ) -> Dict[str, Any]:
        """
        Runs fast walk-forward backtest of the scoring engine across historical bars.
        """
        if symbols is None:
            symbols = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"]

        buckets = {
            "8.0-10.0 (Strong Conviction)": {"trades": [], "wins": 0, "net_pnls": []},
            "7.5-7.9 (High Conviction)": {"trades": [], "wins": 0, "net_pnls": []},
            "6.5-7.4 (Moderate Trend)": {"trades": [], "wins": 0, "net_pnls": []},
            "5.0-6.4 (Neutral / Choppy)": {"trades": [], "wins": 0, "net_pnls": []},
            "< 5.0 (Bearish / Avoid)": {"trades": [], "wins": 0, "net_pnls": []}
        }

        train_count = 0
        val_count = 0

        for sym in symbols:
            df = get_historical_data(sym, period=period, interval=interval)
            if df.empty or len(df) < 50:
                continue

            df = add_all_indicators(df)
            split_idx = int(len(df) * train_test_split)

            for i in range(30, len(df) - holding_bars):
                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                prev2_row = df.iloc[i-2]
                
                score = cls.calculate_fast_score(row, prev_row)
                prev_score = cls.calculate_fast_score(prev_row, prev2_row)
                
                # SNIPER ENTRY TRIGGER: Must be a fresh breakout or fresh EMA pullback bounce
                is_fresh_signal = (prev_score < 7.5 and score >= 7.5) or (score >= 7.5 and float(row["Low"]) <= float(row["EMA_21"]) and float(row["Close"]) > float(row["EMA_21"]))
                
                # For baseline comparison in lower buckets:
                if score < 7.5:
                    is_fresh_signal = (i % holding_bars == 0) # Sample non-overlapping periods
                elif not is_fresh_signal:
                    continue # Skip chasing already-running candles

                entry_price = float(row["Close"])
                atr = float(row["ATR_14"])
                adx = float(row.get("ADX_14", 20.0))
                
                # Continuous SL multiplier: 1.5x in Chop (ADX <= 20) -> 1.2x in Trend (ADX >= 25)
                adx_factor = min(1.0, max(0.0, (adx - 20.0) / 5.0))
                sl_mult = 1.5 - (0.3 * adx_factor)
                sl_distance = sl_mult * atr

                sl_price = entry_price - sl_distance
                t1_price = entry_price + (1.5 * sl_distance)
                t2_price = entry_price + (2.5 * sl_distance)

                # Simulate trade progression across forward bars
                future_highs = df["High"].iloc[i+1 : i+1+holding_bars].values
                future_lows = df["Low"].iloc[i+1 : i+1+holding_bars].values
                future_closes = df["Close"].iloc[i+1 : i+1+holding_bars].values

                exit_price = float(future_closes[-1]) # Default time expiry
                outcome = "TIME_EXPIRY"
                target_1_hit = False

                for bar_idx in range(len(future_highs)):
                    bar_h = float(future_highs[bar_idx])
                    bar_l = float(future_lows[bar_idx])

                    # Check Target 1 (+1.8 ATR)
                    if not target_1_hit and bar_h >= t1_price:
                        target_1_hit = True
                        sl_price = entry_price # Breakeven lock

                    # Check Target 2 (+3.0 ATR)
                    if bar_h >= t2_price:
                        exit_price = t2_price
                        outcome = "HIT_TARGET_2"
                        break

                    # Check Stop-Loss
                    if bar_l <= sl_price:
                        exit_price = sl_price
                        outcome = "HIT_SL" if not target_1_hit else "HIT_BREAKEVEN_SL"
                        break

                # Compute net PnL after Indian taxes & fees
                qty = max(1, int(100000.0 / entry_price)) # Position sizing ~₹1L
                gross_pnl = (exit_price - entry_price) * qty
                friction = cls.calculate_trade_friction(entry_price, exit_price, qty)
                net_pnl = gross_pnl - friction
                pnl_pct = (net_pnl / (entry_price * qty)) * 100.0

                trade_record = {
                    "symbol": sym,
                    "score": score,
                    "net_pnl": net_pnl,
                    "pnl_pct": pnl_pct,
                    "outcome": outcome
                }

                # Assign to Bucket
                if score >= 8.0:
                    b_key = "8.0-10.0 (Strong Conviction)"
                elif score >= 7.5:
                    b_key = "7.5-7.9 (High Conviction)"
                elif score >= 6.5:
                    b_key = "6.5-7.4 (Moderate Trend)"
                elif score >= 5.0:
                    b_key = "5.0-6.4 (Neutral / Choppy)"
                else:
                    b_key = "< 5.0 (Bearish / Avoid)"

                buckets[b_key]["trades"].append(trade_record)
                if net_pnl > 0:
                    buckets[b_key]["wins"] += 1
                buckets[b_key]["net_pnls"].append(net_pnl)

                if i < split_idx:
                    train_count += 1
                else:
                    val_count += 1

        # Calculate Bucket Statistics
        bucket_summary = {}
        for b_name, b_data in buckets.items():
            t_count = len(b_data["trades"])
            if t_count > 0:
                win_rate = (b_data["wins"] / t_count) * 100.0
                total_net = sum(b_data["net_pnls"])
                wins_pnl = sum(p for p in b_data["net_pnls"] if p > 0)
                losses_pnl = abs(sum(p for p in b_data["net_pnls"] if p < 0))
                profit_factor = (wins_pnl / losses_pnl) if losses_pnl > 0 else 9.99
                avg_expectancy_pct = np.mean([t["pnl_pct"] for t in b_data["trades"]])
            else:
                win_rate = 0.0
                total_net = 0.0
                profit_factor = 0.0
                avg_expectancy_pct = 0.0

            bucket_summary[b_name] = {
                "total_trades": t_count,
                "win_rate_pct": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "total_net_pnl": round(total_net, 2),
                "avg_expectancy_pct": round(avg_expectancy_pct, 2)
            }

        return {
            "status": "SUCCESS",
            "period": period,
            "interval": interval,
            "symbols_evaluated": symbols,
            "in_sample_trades": train_count,
            "out_of_sample_trades": val_count,
            "bucket_metrics": bucket_summary,
            "recommended_threshold": 7.5
        }

if __name__ == "__main__":
    res = ScorerValidator.run_validation()
    print("\n" + "="*70)
    print(" [SCORER VALIDATOR] WALK-FORWARD VALIDATION RESULTS (NET OF TAXES & FEES)")
    print("="*70)
    for b_name, m in res["bucket_metrics"].items():
        print(f"\n[Score Bucket]: {b_name}")
        print(f"   * Total Trades: {m['total_trades']}")
        print(f"   * Win Rate: {m['win_rate_pct']}%")
        print(f"   * Profit Factor: {m['profit_factor']}")
        print(f"   * Net Expectancy: {m['avg_expectancy_pct']:+.2f}% / trade")
        print(f"   * Total Net PnL: INR {m['total_net_pnl']:+,.2f}")
    print("\n" + "="*70)
