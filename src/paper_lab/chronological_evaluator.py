"""
Chronological Bar-by-Bar Replay Engine for Paper Lab Outcome Resolution.
Replays 1-minute (or 5-minute) OHLC candles to determine true first-hit level (SL vs T1 vs T2 vs EOD Close).
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.paper_lab.paper_db import PaperDB
from src.paper_lab.lab_config import LabConfig
from src.data.data_fetcher import get_historical_data, get_live_quote
from src.utils.helpers import clean_symbol, get_ist_now

class ChronologicalEvaluator:
    """
    Evaluates intraday trade outcomes by stepping chronologically through 1m/5m price bars.
    Eliminates sampling error and accurately resolves whether SL or Target was hit first.
    """

    @classmethod
    def evaluate_pick_candles(
        cls,
        pick: Dict[str, Any],
        df_candles: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single pick against an intraday DataFrame of OHLC candles.
        """
        symbol = pick["symbol"]
        entry_p = float(pick.get("entry_price") or pick.get("signal_price", 100.0))
        t1_p = float(pick["target_1"])
        t2_p = float(pick["target_2"])
        sl_p = float(pick["stop_loss"])
        qty = int(pick.get("quantity", 1))
        cap = float(pick.get("allocated_capital", 20000.0))
        pick_date = pick["pick_date"]

        # 1. Fetch candles if not supplied
        if df_candles is None or df_candles.empty:
            resolution_method = "1M_CANDLE_REPLAY"
            df = get_historical_data(symbol, period="1d", interval="1m")
            if df.empty or len(df) < 5:
                # Fallback to 5m interval
                df = get_historical_data(symbol, period="1d", interval="5m")
                resolution_method = "5M_CANDLE_REPLAY" if not df.empty else "EOD_FALLBACK"
        else:
            df = df_candles
            resolution_method = "1M_CANDLE_REPLAY"

        # Fallback if no intraday candle data is available
        if df is None or df.empty or len(df) < 2:
            quote = get_live_quote(symbol)
            curr_p = float(quote.get("price", entry_p))
            pnl_rs = round((curr_p - entry_p) * qty, 2)
            pnl_pct = round(((curr_p - entry_p) / entry_p) * 100.0, 2)
            return {
                "pick_date": pick_date,
                "symbol": symbol,
                "entry_price": entry_p,
                "exit_price": curr_p,
                "exit_time": "15:25:00",
                "exit_type": "EOD_CLOSE",
                "pnl_rs": pnl_rs,
                "pnl_pct": pnl_pct,
                "allocated_capital": cap,
                "quantity": qty,
                "max_favorable_excursion_rs": max(0.0, pnl_rs),
                "max_adverse_excursion_rs": min(0.0, pnl_rs),
                "bars_held": 1,
                "resolution_method": "EOD_FALLBACK"
            }

        # 2. Chronological Replay
        peak_high = entry_p
        trough_low = entry_p
        exit_type = "EOD_CLOSE"
        exit_price = float(df["Close"].iloc[-1])
        exit_time = str(df.index[-1]).split(" ")[-1] if " " in str(df.index[-1]) else "15:25:00"
        bars_held = len(df)

        for i, (idx, bar) in enumerate(df.iterrows()):
            b_open = float(bar["Open"])
            b_high = float(bar["High"])
            b_low = float(bar["Low"])
            b_close = float(bar["Close"])
            b_time = str(idx).split(" ")[-1] if " " in str(idx) else str(idx)

            if b_high > peak_high:
                peak_high = b_high
            if b_low < trough_low:
                trough_low = b_low

            hit_sl = (b_low <= sl_p)
            hit_t1 = (b_high >= t1_p)
            hit_t2 = (b_high >= t2_p)

            # Case A: Same-bar dual touch (rare edge case)
            if hit_sl and (hit_t1 or hit_t2):
                # Resolve based on open price proximity and candle polarity
                dist_open_to_sl = abs(b_open - sl_p)
                dist_open_to_t = abs(b_open - t1_p)
                is_red = (b_close < b_open)

                if is_red or dist_open_to_sl < dist_open_to_t:
                    exit_type = "SL_HIT"
                    exit_price = sl_p if b_open >= sl_p else b_open  # Account for gap below SL
                else:
                    exit_type = "T2_HIT" if hit_t2 else "T1_HIT"
                    exit_price = t2_p if hit_t2 else t1_p

                exit_time = b_time
                bars_held = i + 1
                break

            # Case B: Stop Loss Hit First
            if hit_sl:
                exit_type = "SL_HIT"
                # If market opened/gapped down below SL, execute at bar open (slippage reality)
                exit_price = sl_p if b_open >= sl_p else b_open
                exit_time = b_time
                bars_held = i + 1
                break

            # Case C: Target 2 Hit First
            if hit_t2:
                exit_type = "T2_HIT"
                exit_price = t2_p if b_open <= t2_p else b_open  # Account for gap above target
                exit_time = b_time
                bars_held = i + 1
                break

            # Case D: Target 1 Hit First
            if hit_t1:
                exit_type = "T1_HIT"
                exit_price = t1_p if b_open <= t1_p else b_open
                exit_time = b_time
                bars_held = i + 1
                break

        # If loop completed without SL or Targets hit:
        if exit_type == "EOD_CLOSE":
            exit_price = float(df["Close"].iloc[-1])
            exit_time = str(df.index[-1]).split(" ")[-1] if " " in str(df.index[-1]) else "15:25:00"
            bars_held = len(df)

        pnl_rs = round((exit_price - entry_p) * qty, 2)
        pnl_pct = round(((exit_price - entry_p) / entry_p) * 100.0, 2)
        mfe_rs = round((peak_high - entry_p) * qty, 2)
        mae_rs = round((trough_low - entry_p) * qty, 2)

        return {
            "pick_date": pick_date,
            "symbol": symbol,
            "entry_price": entry_p,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "exit_type": exit_type,
            "pnl_rs": pnl_rs,
            "pnl_pct": pnl_pct,
            "allocated_capital": cap,
            "quantity": qty,
            "max_favorable_excursion_rs": max(0.0, mfe_rs),
            "max_adverse_excursion_rs": min(0.0, mae_rs),
            "bars_held": bars_held,
            "resolution_method": resolution_method
        }

    @classmethod
    def evaluate_all_picks_for_date(cls, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Runs at 03:35 PM IST:
        Fetches all unresolved picks for the date, performs chronological replay, and saves outcomes.
        """
        now = get_ist_now()
        date_str = target_date or now.strftime("%Y-%m-%d")

        picks = PaperDB.get_picks_by_date(date_str)
        if not picks:
            print(f"[ChronologicalEvaluator] No picks found for {date_str}.")
            return []

        print(f"[ChronologicalEvaluator] Evaluating {len(picks)} picks for {date_str} via Chronological 1m/5m Candle Replay...")
        outcomes = []

        for p in picks:
            outcome = cls.evaluate_pick_candles(p)
            PaperDB.save_outcome(outcome)
            outcomes.append(outcome)

            icon = "[WIN]" if outcome["exit_type"].startswith("T") else ("[LOSS]" if outcome["exit_type"].startswith("SL") else "[EOD]")
            print(f"  {icon} {p['symbol']}: {outcome['exit_type']} @ Rs {outcome['exit_price']:,.2f} ({outcome['exit_time']}) | P&L: Rs {outcome['pnl_rs']:+,.2f} ({outcome['pnl_pct']:+.2f}%)")

        return outcomes
