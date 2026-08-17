"""
Smart Stock Advisor & Quantitative Scoring Engine for Indian Equities & Indices.
Implements:
1. Orthogonal Category Score Capping (Eliminates indicator redundancy).
2. ADX-Based Regime Filtering (Trending vs. Range-Bound Chop).
3. Dynamic ATR-Derived Targets (Replaces arbitrary flat percentages).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.data.data_fetcher import get_historical_data, get_live_quote
from src.strategies.indicators import (
    calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_supertrend,
    calculate_adx, calculate_vwap, add_all_indicators
)
from src.utils.helpers import clean_symbol, display_symbol_name, format_currency_inr

class StockAdvisor:
    """
    Quantitative scoring and trade setup engine.
    """

    @classmethod
    def evaluate_df_slice(cls, df: pd.DataFrame, symbol: str = "ASSET") -> Dict[str, Any]:
        """
        Evaluates a slice of historical candles (strictly closed bars) using
        orthogonal category capping and ADX regime filters.
        """
        if df.empty or len(df) < 25:
            return {"status": "ERROR", "score": 5.0, "message": "Insufficient data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
        
        curr_p = float(close.iloc[-1])
        prev_p = float(close.iloc[-2]) if len(close) > 1 else curr_p
        
        # Calculate Indicators (Ensuring all are available)
        ema9 = float(calculate_ema(close, 9).iloc[-1])
        ema21 = float(calculate_ema(close, 21).iloc[-1])
        ema50 = float(calculate_ema(close, 50).iloc[-1]) if len(df) >= 50 else ema21
        ema200 = float(calculate_ema(close, 200).iloc[-1]) if len(df) >= 200 else ema50
        
        rsi = float(calculate_rsi(close, 14).iloc[-1])
        macd, macd_sig, macd_hist = calculate_macd(close, 12, 26, 9)
        last_hist = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else last_hist
        
        st, st_dir = calculate_supertrend(high, low, close, 10, 3.0)
        last_st_dir = int(st_dir.iloc[-1])
        
        atr = round(float(calculate_atr(high, low, close, 14).iloc[-1]), 2)
        adx_s, p_di, m_di = calculate_adx(high, low, close, 14)
        last_adx = float(adx_s.iloc[-1])
        last_pdi = float(p_di.iloc[-1])
        last_mdi = float(m_di.iloc[-1])
        
        vwap_val = float(calculate_vwap(high, low, close, volume).iloc[-1])
        
        # 1. REGIME DETECTION (Continuous ADX Multiplier)
        # mu(ADX) smoothly scales from 0.5 (Chop <= 20) to 1.0 (Trend >= 25)
        adx_factor = min(1.0, max(0.0, (last_adx - 20.0) / 5.0))
        mu_trend = 0.5 + 0.5 * adx_factor

        if last_adx >= 25.0:
            regime = "TRENDING"
            regime_desc = f"Strong directional trend active (ADX: {last_adx:.1f})"
        elif last_adx <= 20.0:
            regime = "RANGE_BOUND"
            regime_desc = f"Choppy consolidation / range-bound market (ADX: {last_adx:.1f})"
        else:
            regime = "TRANSITIONAL"
            regime_desc = f"Developing momentum (ADX: {last_adx:.1f}, Scaling Factor: {mu_trend:.2f}x)"

        pros = []
        watchouts = []

        # =========================================================================
        # BUCKET 1: TREND ALIGNMENT (Exact Max +2.50 / Min -2.50 pts)
        # =========================================================================
        trend_pts = 0.0
        if ema9 > ema21:
            trend_pts += 0.75
        elif ema9 < ema21:
            trend_pts -= 0.75

        if curr_p > ema50:
            trend_pts += 0.75
        elif curr_p < ema50:
            trend_pts -= 0.75

        if curr_p > ema200:
            trend_pts += 0.50
        elif curr_p < ema200:
            trend_pts -= 0.50

        if last_st_dir == 1:
            trend_pts += 0.50
        elif last_st_dir == -1:
            trend_pts -= 0.50
            
        # Apply continuous regime multiplier symmetrically to trend score
        trend_pts = trend_pts * mu_trend
        if mu_trend < 1.0:
            if regime == "RANGE_BOUND":
                watchouts.append("⚠️ **Range-Bound Market:** Trend signals scaled by 0.50x to avoid false breakouts.")
            else:
                watchouts.append(f"⚠️ **Transitional Regime:** Trend signals scaled by {mu_trend:.2f}x.")
        else:
            if trend_pts >= 1.5:
                pros.append("🟢 **Confirmed Trend Alignment:** Price trades cleanly above key institutional EMAs.")

        trend_pts = max(-2.5, min(2.5, trend_pts))

        # =========================================================================
        # BUCKET 2: MOMENTUM & RELATIVE STRENGTH (Max Capped at +2.0 / -2.0 pts)
        # =========================================================================
        mom_pts = 0.0
        if 50.0 <= rsi <= 68.0:
            mom_pts += 1.0
            pros.append(f"🟢 **Optimal Buyer Energy:** RSI at {rsi:.1f} (Sweet spot with room to run).")
        elif rsi > 70.0:
            mom_pts -= 0.5
            watchouts.append(f"⚠️ **Overheated:** RSI at {rsi:.1f} (Consider waiting for a pullback).")
        elif rsi < 35.0:
            # Mean-reversion bonus in range-bound market, penalty in trending
            if regime == "RANGE_BOUND":
                mom_pts += 1.0
                pros.append(f"⚡ **Range Bounce Zone:** Oversold RSI at {rsi:.1f} inside support range.")
            else:
                mom_pts -= 0.5
                watchouts.append(f"🔴 **Heavy Selling Pressure:** RSI at {rsi:.1f}.")

        if last_hist > 0 and last_hist > prev_hist:
            mom_pts += 1.0
            pros.append("🟢 **Expanding Velocity:** MACD histogram accelerating green.")
        elif last_hist < 0:
            mom_pts -= 0.5

        mom_pts = max(-2.0, min(2.0, mom_pts))

        # =========================================================================
        # BUCKET 3: VOLATILITY & LOCATION (Max Capped at +1.5 / -1.5 pts)
        # =========================================================================
        vol_loc_pts = 0.0
        ub, mb, lb, bw, pct_b = calculate_bollinger_bands(close, 20, 2.0)
        last_pct_b = float(pct_b.iloc[-1])
        last_bw = float(bw.iloc[-1])

        if 0.4 <= last_pct_b <= 0.8:
            vol_loc_pts += 0.8 # Healthy mid-to-upper band location
        elif last_pct_b > 0.95:
            vol_loc_pts -= 0.5 # Near extreme upper band
            watchouts.append("⚠️ **Upper Band Tag:** Price pressing the 2-sigma Bollinger ceiling.")
        elif last_pct_b < 0.05:
            if regime == "RANGE_BOUND":
                vol_loc_pts += 0.5 # Lower band bounce support
                pros.append("⚡ **Lower Band Support:** Price holding lower 2-sigma band.")
            else:
                vol_loc_pts -= 0.5 # Lower band breakdown risk
                watchouts.append("🔴 **Lower Band Breakdown:** Price pressing lower 2-sigma floor.")

        # Squeeze expansion
        if last_bw < 4.0:
            vol_loc_pts += 0.7
            pros.append("⚡ **Volatility Squeeze:** Tight Bollinger Bandwidth indicates impending explosive expansion.")

        vol_loc_pts = max(-1.5, min(1.5, vol_loc_pts))

        # =========================================================================
        # BUCKET 4: VOLUME & INSTITUTIONAL FLOW (Max Capped at +1.5 / -1.5 pts)
        # =========================================================================
        flow_pts = 0.0
        avg_vol = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else 1.0
        curr_vol = float(volume.iloc[-1])
        
        if curr_vol > avg_vol * 1.25:
            flow_pts += 0.8
            pros.append("🟢 **Institutional Volume Surge:** Volume > 1.25x average.")
        elif curr_vol < avg_vol * 0.50:
            flow_pts -= 0.4
            watchouts.append("⚠️ **Low Participation:** Volume is less than 50% of 20-period average.")
            
        if curr_p > vwap_val:
            flow_pts += 0.7
            pros.append(f"🟢 **Above Institutional Benchmark:** Trading above VWAP (₹{vwap_val:.2f}).")
        else:
            flow_pts -= 0.5

        flow_pts = max(-1.5, min(1.5, flow_pts))

        # =========================================================================
        # COMPOSITE SCORE (Baseline 5.0 + Sum of Orthogonal Buckets)
        # =========================================================================
        raw_score = 5.0 + trend_pts + mom_pts + vol_loc_pts + flow_pts
        final_score = round(max(1.0, min(9.8, raw_score)), 1)

        # Verdict
        if final_score >= 7.5:
            verdict = "🟢 STRONG BUY"
            verdict_desc = "High-probability setup with multi-bucket confirmation."
            action = "BUY NOW"
            badge_color = "#3fb950"
        elif final_score >= 6.2:
            verdict = "🟢 BUY ON PULLBACK"
            verdict_desc = "Positive overall structure. Enter on minor pullback to 20 EMA."
            action = "BUY ON DIP"
            badge_color = "#2ea043"
        elif final_score >= 4.5:
            verdict = "🟡 WAIT / NEUTRAL"
            verdict_desc = "Consolidating market structure. Capital preserved."
            action = "WAIT"
            badge_color = "#d29922"
        else:
            verdict = "🔴 AVOID / BEARISH"
            verdict_desc = "Negative structure or heavy selling pressure."
            action = "AVOID"
            badge_color = "#f85149"

        # =========================================================================
        # REGIME-AWARE CONTINUOUS STOP-LOSS & INVARIANT BLENDED TARGETS
        # =========================================================================
        # Continuous SL multiplier: 1.5x in Chop (ADX <= 20) -> 1.2x in Trend (ADX >= 25)
        sl_mult = 1.5 - (0.3 * adx_factor)
        sl_distance = sl_mult * atr
        sl_price = round(curr_p - sl_distance, 2)
        
        # Targets are tied directly to SL distance (1.5x SL and 2.5x SL)
        # This guarantees that Blended R:R (0.5 * 1.5R + 0.5 * 2.5R) = 2.0R across ALL regimes!
        t1_price = round(curr_p + (1.5 * sl_distance), 2)
        t2_price = round(curr_p + (2.5 * sl_distance), 2)

        t1_gain = round(((t1_price - curr_p) / curr_p) * 100.0, 2) if curr_p > 0 else 0.0
        t2_gain = round(((t2_price - curr_p) / curr_p) * 100.0, 2) if curr_p > 0 else 0.0
        sl_loss = round(((curr_p - sl_price) / curr_p) * 100.0, 2) if curr_p > 0 else 0.0
        t1_rr = round(t1_gain / sl_loss, 2) if sl_loss > 0 else 1.5
        t2_rr = round(t2_gain / sl_loss, 2) if sl_loss > 0 else 2.5
        entry_zone_str = f"₹{curr_p * 0.998:.2f} – ₹{curr_p:.2f}"

        return {
            "status": "SUCCESS",
            "symbol": symbol,
            "display_name": display_symbol_name(symbol),
            "current_price": curr_p,
            "score": final_score,
            "regime": regime,
            "regime_description": regime_desc,
            "verdict": verdict,
            "verdict_desc": verdict_desc,
            "action": action,
            "badge_color": badge_color,
            "entry_zone": entry_zone_str,
            "target_1": {
                "price": t1_price,
                "gain_pct": t1_gain,
                "reward_risk": t1_rr
            },
            "target_2": {
                "price": t2_price,
                "gain_pct": t2_gain,
                "reward_risk": t2_rr
            },
            "stop_loss": {
                "price": sl_price,
                "loss_pct": sl_loss
            },
            "buckets": {
                "trend": round(trend_pts, 2),
                "momentum": round(mom_pts, 2),
                "volatility": round(vol_loc_pts, 2),
                "volume_flow": round(flow_pts, 2)
            },
            "metrics": {
                "ema9": round(ema9, 2),
                "ema21": round(ema21, 2),
                "ema50": round(ema50, 2),
                "rsi": round(rsi, 1),
                "adx": round(last_adx, 1),
                "atr": round(atr, 2),
                "vwap": round(vwap_val, 2)
            },
            "levels": {
                "entry_zone": entry_zone_str,
                "stop_loss": sl_price,
                "target_1": t1_price,
                "target_2": t2_price,
                "risk_reward": f"1:{t1_rr} to Target 1 | 1:{t2_rr} to Target 2"
            },
            "pros": pros,
            "watchouts": watchouts
        }

    @classmethod
    def analyze_stock(cls, symbol: str, horizon: str = "swing") -> Dict[str, Any]:
        """
        Analyze stock for given horizon using live data.
        """
        sym = clean_symbol(symbol)
        if horizon == "intraday":
            period = "5d"
            interval = "5m"
            time_text = "30 Mins to 4 Hours (Same Day)"
        elif horizon == "positional":
            period = "1y"
            interval = "1d"
            time_text = "2 to 4 Weeks"
        else: # swing
            period = "1mo"
            interval = "15m"
            time_text = "3 to 7 Trading Days"

        df = get_historical_data(sym, period=period, interval=interval)
        if df.empty or len(df) < 25:
            return {"status": "ERROR", "message": f"Could not load sufficient data for {sym}"}

        res = cls.evaluate_df_slice(df, sym)
        res["horizon"] = horizon
        res["horizon_text"] = time_text
        res["holding_time_text"] = time_text
        return res
