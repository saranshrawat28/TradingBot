"""
Pre-Market Opening Intelligence & Morning Intraday Stock Suggester for Indian Markets (NSE / BSE).
Analyzes:
1. Global cues, index gap direction, and overall opening sentiment (09:00 - 09:15 AM).
2. Liquid Indian equities for morning momentum, volume surges, and breakout potential.
3. Provides Top 3-5 curated, plain-English intraday stock recommendations with exact Entry, Targets in ₹, and Stop-Loss.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from src.data.data_fetcher import get_live_quote, get_historical_data
from src.engine.stock_advisor import StockAdvisor
from src.utils.helpers import clean_symbol, display_symbol_name, get_ist_now, format_currency_inr

# High-liquidity, high-momentum Indian stocks for daily pre-market scanning
DEFAULT_PREMARKET_UNIVERSE = [
    "RELIANCE.NS",
    "TMCV.NS",      # Tata Motors
    "INFY.NS",      # Infosys
    "HDFCBANK.NS",  # HDFC Bank
    "ICICIBANK.NS", # ICICI Bank
    "SBIN.NS",      # State Bank of India
    "TCS.NS",       # Tata Consultancy Services
    "BHARTIARTL.NS",# Bharti Airtel
    "LT.NS",        # Larsen & Toubro
    "ETERNAL.NS",   # Zomato
    "M&M.NS",       # Mahindra & Mahindra
    "SUNPHARMA.NS", # Sun Pharma
    "BAJFINANCE.NS",# Bajaj Finance
    "AXISBANK.NS",  # Axis Bank
    "TITAN.NS"      # Titan
]

class PreMarketAnalyzer:
    """
    Automated Morning Intelligence and Intraday Stock Suggester.
    """

    @classmethod
    def get_market_opening_sentiment(cls) -> Dict[str, Any]:
        """
        Determines expected market opening direction and mood based on Nifty 50, Bank Nifty, and VIX.
        """
        nifty_quote = get_live_quote("^NSEI")
        banknifty_quote = get_live_quote("^NSEBANK")
        vix_quote = get_live_quote("^INDIAVIX")

        nifty_p = float(nifty_quote.get("price", 24350.0))
        nifty_prev = float(nifty_quote.get("previous_close", nifty_p))
        nifty_chg_pct = float(nifty_quote.get("change_pct", 0.0))
        
        # Calculate gap pct
        if nifty_prev > 0:
            gap_pct = round(((nifty_p - nifty_prev) / nifty_prev) * 100.0, 2)
        else:
            gap_pct = nifty_chg_pct

        vix_level = float(vix_quote.get("price", 13.5))

        # Determine Market Session Timing
        now_ist = get_ist_now()
        hour_minute = now_ist.hour * 100 + now_ist.minute

        if 900 <= hour_minute < 908:
            market_phase = "PRE_OPEN_COLLECTION"
            phase_desc = "NSE Pre-Open Order Collection Phase (09:00 - 09:08 AM)"
        elif 908 <= hour_minute < 915:
            market_phase = "PRE_OPEN_MATCHING"
            phase_desc = "Pre-Open Price Matching & Discovery Phase (09:08 - 09:15 AM)"
        elif 915 <= hour_minute <= 1530:
            market_phase = "LIVE_MARKET"
            phase_desc = "Regular Live Market Trading Session (09:15 AM - 03:30 PM)"
        else:
            market_phase = "MARKET_CLOSED"
            phase_desc = "Market Closed (Showing Pre-Market Opening Setup for Next Session)"

        # Classify Opening Mood
        if gap_pct >= 0.40:
            sentiment = "BULLISH_GAP_UP"
            title = "🟢 BULLISH OPEN EXPECTED (Gap Up)"
            badge_color = "#10b981"
            explanation = f"NIFTY 50 is set to open strong (+{gap_pct:.2f}%). Buyers are aggressive in early morning cues. Focus on buying top momentum stocks on opening dips."
        elif gap_pct <= -0.40:
            sentiment = "BEARISH_GAP_DOWN"
            title = "🔴 BEARISH OPEN EXPECTED (Gap Down)"
            badge_color = "#f43f5e"
            explanation = f"NIFTY 50 is opening weak ({gap_pct:.2f}%). Heavy global selling pressure observed. Avoid hasty morning buys and wait for support to hold."
        else:
            sentiment = "FLAT_NEUTRAL"
            title = "🟡 FLAT / BALANCED OPEN EXPECTED"
            badge_color = "#f59e0b"
            explanation = f"NIFTY 50 is opening flat ({gap_pct:+.2f}%). Market is consolidating. Focus on stock-specific breakouts rather than index direction."

        return {
            "sentiment": sentiment,
            "title": title,
            "badge_color": badge_color,
            "explanation": explanation,
            "gap_pct": gap_pct,
            "nifty_price": nifty_p,
            "banknifty_price": float(banknifty_quote.get("price", 51200.0)),
            "vix_level": vix_level,
            "market_phase": market_phase,
            "phase_description": phase_desc,
            "timestamp": now_ist.strftime("%I:%M %p IST, %d %b %Y")
        }

    @classmethod
    def scan_pre_market_stocks(
        cls,
        universe: Optional[List[str]] = None,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Scans liquid Indian stocks and generates curated, plain-English intraday recommendations.
        """
        symbols = universe or DEFAULT_PREMARKET_UNIVERSE
        scanned_items = []
        gap_ups = []
        gap_downs = []

        # Market sentiment
        opening_info = cls.get_market_opening_sentiment()
        index_trend = "BULLISH" if "BULLISH" in opening_info["sentiment"] else ("BEARISH" if "BEARISH" in opening_info["sentiment"] else "NEUTRAL")

        for sym in symbols:
            try:
                df = get_historical_data(sym, period="5d", interval="5m")
                if df.empty or len(df) < 25:
                    continue

                quote = get_live_quote(sym)
                curr_p = float(quote.get("price", df["Close"].iloc[-1]))
                prev_p = float(quote.get("previous_close", df["Close"].iloc[-2] if len(df) > 1 else curr_p))
                gap_pct = round(((curr_p - prev_p) / prev_p) * 100.0, 2) if prev_p > 0 else 0.0

                analysis = StockAdvisor.evaluate_df_slice(df, symbol=sym, horizon="intraday", index_trend=index_trend)
                score = float(analysis.get("score", 5.0))
                verdict = analysis.get("verdict", "WAIT")
                t1 = analysis.get("target_1", {})
                t2 = analysis.get("target_2", {})
                sl = analysis.get("stop_loss", {})
                levels = analysis.get("levels", {})

                disp_name = display_symbol_name(sym)

                # Track Gap Ups / Gap Downs
                item_summary = {
                    "symbol": sym,
                    "name": disp_name,
                    "price": curr_p,
                    "gap_pct": gap_pct,
                    "score": score
                }
                if gap_pct >= 0.8:
                    gap_ups.append(item_summary)
                elif gap_pct <= -0.8:
                    gap_downs.append(item_summary)

                # Formulate Plain-English Reasons
                if score >= 7.5:
                    action = "BUY"
                    action_title = "🟢 STRONG BUY"
                    action_badge = "#10b981"
                    reason_text = f"Strong buyer momentum! Price is holding firmly above key support levels with positive volume flow. High probability of up-move towards Target 1."
                elif score >= 6.2:
                    action = "BUY_ON_DIP"
                    action_title = "🟢 BUY ON DIP"
                    action_badge = "#22c55e"
                    reason_text = f"Positive upward structure. Best to buy when price pulls back slightly to the entry zone."
                elif score <= 4.0:
                    action = "SELL"
                    action_title = "🔴 SELL / SHORT"
                    action_badge = "#f43f5e"
                    reason_text = f"Heavy selling pressure observed. Structure is breaking down below session resistance."
                else:
                    action = "WAIT"
                    action_title = "🟡 WAIT / WATCH"
                    action_badge = "#f59e0b"
                    reason_text = f"Stock is moving sideways in a consolidation range. Better to wait for a clear breakout before entering."

                t1_p = float(t1.get("price", curr_p * 1.025))
                t2_p = float(t2.get("price", curr_p * 1.045))
                sl_p = float(sl.get("price", curr_p * 0.985))

                t1_gain_pct = round(((t1_p - curr_p) / curr_p) * 100.0, 2)
                sl_loss_pct = round(((curr_p - sl_p) / curr_p) * 100.0, 2)

                scanned_items.append({
                    "symbol": sym,
                    "display_name": disp_name,
                    "current_price": curr_p,
                    "gap_pct": gap_pct,
                    "score": score,
                    "action": action,
                    "action_title": action_title,
                    "action_badge": action_badge,
                    "entry_zone": f"₹{curr_p * 0.998:,.2f} – ₹{curr_p:,.2f}",
                    "target_1_price": t1_p,
                    "target_1_gain_pct": t1_gain_pct,
                    "target_2_price": t2_p,
                    "target_2_gain_pct": round(((t2_p - curr_p) / curr_p) * 100.0, 2),
                    "stop_loss_price": sl_p,
                    "stop_loss_pct": sl_loss_pct,
                    "risk_reward": f"1:{round(t1_gain_pct / max(0.1, sl_loss_pct), 1)} to Target 1",
                    "reason": reason_text,
                    "pros": analysis.get("pros", [])[:2],
                    "watchouts": analysis.get("watchouts", [])[:1]
                })

            except Exception:
                continue

        # Sort scanned items by score (highest confidence first)
        scanned_items.sort(key=lambda x: x["score"], reverse=True)
        top_picks = scanned_items[:top_n]

        # Sort gap leaderboards
        gap_ups.sort(key=lambda x: x["gap_pct"], reverse=True)
        gap_downs.sort(key=lambda x: x["gap_pct"])

        return {
            "opening_sentiment": opening_info,
            "top_picks": top_picks,
            "all_scanned": scanned_items,
            "top_gap_ups": gap_ups[:5],
            "top_gap_downs": gap_downs[:5],
            "scanned_count": len(scanned_items),
            "generated_at": get_ist_now().strftime("%I:%M:%S %p IST")
        }
