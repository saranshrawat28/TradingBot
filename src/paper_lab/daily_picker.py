"""
Daily Stock Recommender & Pre-Market Signal Generator for Paper Lab.
Selects Top 5 momentum setups with ₹1,00,000 fixed daily notional allocation (₹20,000 / stock).
"""

from datetime import datetime, date
import time
from typing import Dict, List, Any, Optional
import pandas as pd

from src.paper_lab.lab_config import LabConfig
from src.paper_lab.paper_db import PaperDB
from src.paper_lab.holiday_calendar import is_trading_day
from src.engine.stock_advisor import StockAdvisor
from src.data.data_fetcher import get_live_quote, get_historical_data
from src.utils.helpers import clean_symbol, display_symbol_name, get_ist_now

class DailyPicker:
    """Automated stock picker and market-open fill executor for Paper Lab."""

    @classmethod
    def generate_morning_signals(cls, target_date: Optional[str] = None, force: bool = False) -> List[Dict[str, Any]]:
        """
        Phase 1 (08:50 AM IST):
        Scans liquid universe, selects Top 5 candidates, stores as PENDING_OPEN.
        """
        now = get_ist_now()
        date_str = target_date or now.strftime("%Y-%m-%d")

        # 1. Holiday Check (unless forced)
        if not force:
            is_trade, reason = is_trading_day(now if not target_date else datetime.strptime(date_str, "%Y-%m-%d"))
            if not is_trade:
                print(f"[DailyPicker] Skipping signal generation: {reason}")
                return []

        # 2. Check if picks already exist today (Idempotency)
        existing_picks = PaperDB.get_picks_by_date(date_str)
        if len(existing_picks) >= LabConfig.PICKS_PER_DAY:
            print(f"[DailyPicker] {len(existing_picks)} picks already exist for {date_str}. Skipping re-generation.")
            return existing_picks

        print(f"[DailyPicker] Scanning {len(LabConfig.UNIVERSE)} stocks for {date_str} Pre-Market Momentum...")
        candidates = []

        for symbol in LabConfig.UNIVERSE:
            try:
                analysis = StockAdvisor.analyze_stock(symbol, horizon="intraday")
                if not analysis or analysis.get("status") == "ERROR":
                    continue

                score = float(analysis.get("score", 0.0))
                verdict = analysis.get("verdict", "NEUTRAL").upper()
                action = analysis.get("action", "HOLD").upper()

                if score >= LabConfig.MIN_ADVISOR_SCORE and ("BUY" in verdict or "BUY" in action):
                    curr_price = float(analysis.get("current_price", 0.0))
                    if curr_price <= 0:
                        continue

                    # Extract indicators for signal failure diagnostics
                    metrics = analysis.get("metrics", {})
                    buckets = analysis.get("buckets", {})
                    vwap_info = analysis.get("vwap_structure", {})
                    pros = analysis.get("pros", [])
                    grade = analysis.get("setup_grade_title", "GRADE A")

                    t1_data = analysis.get("target_1", {})
                    t2_data = analysis.get("target_2", {})
                    sl_data = analysis.get("stop_loss", {})

                    t1_p = float(t1_data.get("price", curr_price * (1 + LabConfig.DEFAULT_T1_GAIN_PCT / 100.0)))
                    t2_p = float(t2_data.get("price", curr_price * (1 + LabConfig.DEFAULT_T2_GAIN_PCT / 100.0)))
                    sl_p = float(sl_data.get("price", curr_price * (1 - LabConfig.DEFAULT_SL_LOSS_PCT / 100.0)))

                    score_breakdown = {
                        "score": score,
                        "rsi": float(metrics.get("rsi", 50.0)),
                        "adx": float(metrics.get("adx", 20.0)),
                        "atr": float(metrics.get("atr", 0.0)),
                        "rvol": float(metrics.get("rvol", 1.0)),
                        "vwap": float(metrics.get("vwap", curr_price)),
                        "vwap_sigma_dist": float(vwap_info.get("dist_sigma", 0.0) if isinstance(vwap_info, dict) else 0.0),
                        "trend_bucket": float(buckets.get("trend", 0.0)),
                        "mom_bucket": float(buckets.get("momentum", 0.0)),
                        "vol_bucket": float(buckets.get("volatility", 0.0)),
                        "flow_bucket": float(buckets.get("volume_flow", 0.0)),
                        "regime": analysis.get("regime", "NORMAL")
                    }

                    candidates.append({
                        "pick_date": date_str,
                        "symbol": symbol,
                        "display_name": display_symbol_name(symbol),
                        "signal_time": now.strftime("%H:%M:%S"),
                        "signal_price": curr_price,
                        "entry_time": None,
                        "entry_price": None,
                        "target_1": t1_p,
                        "target_2": t2_p,
                        "stop_loss": sl_p,
                        "allocated_capital": LabConfig.DAILY_CAPITAL_PER_PICK,
                        "quantity": max(1, int(LabConfig.DAILY_CAPITAL_PER_PICK / curr_price)),
                        "advisor_score": score,
                        "setup_grade": grade,
                        "score_breakdown": score_breakdown,
                        "top_signals": pros[:4] if pros else ["Strong intraday quantitative score."],
                        "config_version": LabConfig.CONFIG_VERSION,
                        "data_stale_flag": 0,
                        "status": "PENDING_OPEN"
                    })
            except Exception as e:
                print(f"[DailyPicker] Error evaluating {symbol}: {e}")

        # 3. Sort by score descending and take Top N
        candidates.sort(key=lambda x: x["advisor_score"], reverse=True)
        top_picks = candidates[:LabConfig.PICKS_PER_DAY]

        if top_picks:
            PaperDB.save_pending_picks(top_picks)
            print(f"[DailyPicker] Successfully saved Top {len(top_picks)} candidates for {date_str}:")
            for i, p in enumerate(top_picks, 1):
                print(f"  {i}. {p['symbol']} | Score: {p['advisor_score']:.1f} | Signal: Rs {p['signal_price']:,.2f} | T1: Rs {p['target_1']:,.2f} | SL: Rs {p['stop_loss']:,.2f}")
        else:
            print(f"[DailyPicker] No candidate stocks met the minimum score threshold ({LabConfig.MIN_ADVISOR_SCORE}) today.")

        return top_picks

    @classmethod
    def confirm_market_open_fills(cls, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Phase 2 (09:15 AM IST):
        Reads PENDING_OPEN picks, fetches real 09:15 AM market open price, updates entry price,
        re-calibrates targets & stop-loss, and sets status to ACTIVE.
        """
        now = get_ist_now()
        date_str = target_date or now.strftime("%Y-%m-%d")

        pending = PaperDB.get_pending_picks(date_str)
        if not pending:
            return []

        print(f"[DailyPicker] Confirming 09:15 AM Market Open Fills for {len(pending)} picks on {date_str}...")
        filled_picks = []

        for pick in pending:
            sym = pick["symbol"]
            quote = get_live_quote(sym)
            open_price = float(quote.get("price", pick.get("signal_price", 100.0)))
            if open_price <= 0:
                open_price = float(pick.get("signal_price", 100.0))

            # Recalibrate Targets & SL based on actual open price
            sig_price = float(pick.get("signal_price", open_price))
            ratio = (open_price / sig_price) if sig_price > 0 else 1.0

            # Scale target and SL proportionally from the real fill
            t1_p = round(float(pick["target_1"]) * ratio, 2)
            t2_p = round(float(pick["target_2"]) * ratio, 2)
            sl_p = round(float(pick["stop_loss"]) * ratio, 2)
            qty = max(1, int(LabConfig.DAILY_CAPITAL_PER_PICK / open_price))

            fill_time = now.strftime("%H:%M:%S")

            PaperDB.update_pick_fill(
                pick_date=date_str,
                symbol=sym,
                entry_price=open_price,
                entry_time=fill_time,
                target_1=t1_p,
                target_2=t2_p,
                stop_loss=sl_p,
                quantity=qty
            )

            pick["entry_price"] = open_price
            pick["entry_time"] = fill_time
            pick["target_1"] = t1_p
            pick["target_2"] = t2_p
            pick["stop_loss"] = sl_p
            pick["quantity"] = qty
            pick["status"] = "ACTIVE"
            filled_picks.append(pick)
            print(f"  > Filled {sym} @ Rs {open_price:,.2f} | Qty: {qty} | T1: Rs {t1_p:,.2f} | SL: Rs {sl_p:,.2f}")

        return filled_picks

    @classmethod
    def run_daily_picker_catchup(cls, target_date: Optional[str] = None, force: bool = False) -> List[Dict[str, Any]]:
        """
        Catch-up / On-Demand helper:
        If no picks exist today, runs signal generation and immediate fill.
        """
        now = get_ist_now()
        date_str = target_date or now.strftime("%Y-%m-%d")

        existing = PaperDB.get_picks_by_date(date_str)
        if not existing:
            cls.generate_morning_signals(date_str, force=force)

        # Confirm any remaining pending fills
        filled = cls.confirm_market_open_fills(date_str)
        return PaperDB.get_picks_by_date(date_str)
