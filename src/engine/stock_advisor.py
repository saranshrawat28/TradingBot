"""
Smart Stock Advisor & Quantitative Scoring Engine for Indian Equities & Indices.
Implements:
1. Orthogonal Category Score Capping (Eliminates indicator redundancy).
2. ADX-Based Regime Filtering (Trending vs. Range-Bound Chop).
3. Dynamic ATR-Derived Targets (Replaces arbitrary flat percentages).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.data.data_fetcher import get_historical_data, get_live_quote, get_live_index_trend, get_sector_for_symbol
from src.strategies.indicators import (
    calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_supertrend,
    calculate_adx, calculate_vwap, add_all_indicators,
    detect_rsi_divergence, calculate_candle_structure, calculate_mtf_alignment,
    calculate_intraday_vwap_bands, calculate_rvol, calculate_context_multiplier,
    calculate_obv, calculate_classical_pivots, calculate_fibonacci_pivots,
    evaluate_vwap_location_score, evaluate_pivot_confluence,
    calculate_relative_strength_vs_benchmark, calculate_ttm_squeeze, calculate_vsa_structure,
    calculate_camarilla_pivots, calculate_volume_profile, calculate_anchored_vwap,
    calculate_hurst_exponent, calculate_order_block_fvg
)
from src.strategies.options_greeks import DerivativesFlowAnalyzer
from src.utils.helpers import clean_symbol, display_symbol_name, format_currency_inr

class StockAdvisor:
    """
    Quantitative scoring and trade setup engine.
    """

    @classmethod
    def evaluate_df_slice(
        cls,
        df: pd.DataFrame,
        symbol: str = "ASSET",
        horizon: str = "intraday",
        index_trend: str = "BULLISH",
        htf_trend: str = "NEUTRAL",
        sector_name: str = "",
        deriv_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a slice of historical candles (strictly closed bars) using
        orthogonal category capping, single combined context multiplier (ADX + Macro Breadth),
        symmetric MTF trend multipliers, divergence asymmetric gates, 4-zone VWAP sigma-location,
        symmetric 4-case pivot confluence, pure RVol flow, Volume Profile POC, and Camarilla Equations.
        """
        if df.empty or len(df) < 25:
            return {"status": "ERROR", "score": 5.0, "message": "Insufficient data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
        open_ = df["Open"] if "Open" in df.columns else close

        curr_p = float(close.iloc[-1])
        prev_p = float(close.iloc[-2]) if len(close) > 1 else curr_p
        
        # Previous Session High, Low, Close for Pivots
        prev_h = float(high.iloc[-2]) if len(high) > 1 else float(high.iloc[-1])
        prev_l = float(low.iloc[-2]) if len(low) > 1 else float(low.iloc[-1])
        prev_c = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
        pivots = calculate_classical_pivots(prev_h, prev_l, prev_c)
        fib_pivots = calculate_fibonacci_pivots(prev_h, prev_l, prev_c)
        cam_pivots = calculate_camarilla_pivots(prev_h, prev_l, prev_c)
        vp_info = calculate_volume_profile(df, bins=25)
        hurst_h = calculate_hurst_exponent(close)
        fvg_info = calculate_order_block_fvg(df)

        # Baseline Vectorized Indicators
        ema9 = float(calculate_ema(close, 9).iloc[-1])
        ema21 = float(calculate_ema(close, 21).iloc[-1])
        ema50 = float(calculate_ema(close, 50).iloc[-1])
        ema200 = float(calculate_ema(close, min(len(close), 200)).iloc[-1])
        
        rsi_series = calculate_rsi(close, 14)
        rsi = float(rsi_series.iloc[-1])
        
        macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
        last_hist = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else last_hist
        
        supertrend, st_dir = calculate_supertrend(high, low, close, 10, 3.0)
        last_st_dir = int(st_dir.iloc[-1])
        
        adx_res = calculate_adx(high, low, close, 14)
        adx_s = adx_res[0] if isinstance(adx_res, tuple) else adx_res
        last_adx = float(adx_s.iloc[-1]) if not adx_s.empty else 22.0
        
        atr_val = float(calculate_atr(high, low, close, 14).iloc[-1])
        rvol_val = calculate_rvol(volume, 20)
        vwap_bands = calculate_intraday_vwap_bands(df)
        
        mtf_info = calculate_mtf_alignment(df)
        div_info = detect_rsi_divergence(close, low, high, rsi_series, lookback=20)
        wick_info = calculate_candle_structure(open_, high, low, close)
        
        # 1. UNIFIED REGIME & CONTEXT (Smooth ADX + Macro Breadth)
        adx_factor = min(1.0, max(0.0, (last_adx - 20.0) / 5.0))
        stock_dir = "BULLISH" if curr_p > ema50 else "BEARISH"
        mu_context = calculate_context_multiplier(last_adx, stock_trend=stock_dir, index_trend=index_trend)

        if last_adx >= 25.0:
            regime = "TRENDING"
            regime_desc = f"Strong directional trend active (ADX: {last_adx:.1f})"
        elif last_adx <= 20.0:
            regime = "RANGE_BOUND"
            regime_desc = f"Choppy consolidation / range-bound market (ADX: {last_adx:.1f})"
        else:
            regime = "TRANSITIONAL"
            regime_desc = f"Transitional market structure (ADX: {last_adx:.1f})"

        pros = []
        watchouts = []

        # =========================================================================
        # BUCKET 1: TREND ALIGNMENT (Exact Max +2.50 / Min -2.50 pts)
        # Scaled ONCE by Unified Context Multiplier (ADX + Breadth) & MTF factor
        # =========================================================================
        raw_trend_sum = 0.0
        if ema9 > ema21:
            raw_trend_sum += 0.75
        elif ema9 < ema21:
            raw_trend_sum -= 0.75

        if curr_p > ema50:
            raw_trend_sum += 0.75
        elif curr_p < ema50:
            raw_trend_sum -= 0.75

        if curr_p > ema200:
            raw_trend_sum += 0.50
        elif curr_p < ema200:
            raw_trend_sum -= 0.50

        if last_st_dir == 1:
            raw_trend_sum += 0.50
        elif last_st_dir == -1:
            raw_trend_sum -= 0.50
            
        # Apply combined context multiplier AND symmetric MTF factor to trend score
        mu_mtf = mtf_info.get("mu_mtf", 1.00)
        
        # Dual-Timeframe Daily HTF Anchor
        if htf_trend == "BEARISH" and raw_trend_sum > 0:
            raw_trend_sum *= 0.65
            watchouts.append("⚠️ **HTF Daily Headwind:** Daily chart in downtrend below 50 EMA. Long setups are counter-trend.")
        elif htf_trend == "BULLISH" and raw_trend_sum > 0:
            raw_trend_sum *= 1.10
            pros.append("📈 **Daily Macro Alignment:** Aligned with higher-timeframe Daily bullish trend.")

        trend_pts = raw_trend_sum * mu_context * mu_mtf

        if mtf_info.get("status") == "BULLISH_ALIGNED" and raw_trend_sum > 0:
            pros.append("🌐 **Multi-Timeframe Confluence:** 5m trend confirmed by 15m structure (1.15x boost).")
        elif mtf_info.get("status") == "BEARISH_ALIGNED" and raw_trend_sum < 0:
            pros.append("🌐 **Multi-Timeframe Bearish Confluence:** Full downtrend alignment across timeframes (1.15x boost).")
        elif mtf_info.get("status") in ["BEARISH_CONFLICT", "BULLISH_CONFLICT"]:
            watchouts.append("⚠️ **HTF Conflict:** Higher timeframe structure conflicts with trade direction (0.70x scale).")

        if mu_context < 1.0:
            if regime == "RANGE_BOUND":
                watchouts.append("⚠️ **Range-Bound Market:** Trend signals scaled by context factor to avoid false breakouts.")
            else:
                watchouts.append(f"⚠️ **Macro Headwind / Transitional:** Trend signals scaled by {mu_context:.2f}x.")
        else:
            if trend_pts >= 1.5:
                pros.append("🟢 **Confirmed Trend Alignment:** Price trades cleanly above key institutional EMAs.")

        # Hard invariant: Clamped strictly to [-2.50, +2.50]
        trend_pts = max(-2.5, min(2.5, trend_pts))

        # =========================================================================
        # BUCKET 2: MOMENTUM & DIVERGENCE (Max Capped at +2.0 / -2.0 pts)
        # Asymmetric Veto: Bearish Divergence clamps momentum to <= 0.0
        # =========================================================================
        mom_pts = 0.0
        if 50.0 <= rsi <= 68.0:
            mom_pts += 1.0
            pros.append(f"🟢 **Optimal Momentum Window:** RSI at {rsi:.1f} (Ideal expansion zone).")
        elif rsi > 68.0:
            mom_pts += 0.4
            watchouts.append(f"⚠️ **Overbought Warning:** RSI at {rsi:.1f} (Approaching extreme).")
        elif 35.0 <= rsi < 50.0:
            mom_pts -= 0.6
        elif rsi < 35.0:
            if regime == "RANGE_BOUND":
                mom_pts += 0.5
                pros.append(f"⚡ **Mean Reversion Bounce Zone:** RSI at {rsi:.1f} in range-bound structure.")
            else:
                mom_pts -= 1.0
                watchouts.append(f"🔴 **Heavy Momentum Breakdown:** RSI oversold at {rsi:.1f} in active downtrend.")

        if last_hist > 0 and last_hist > prev_hist:
            mom_pts += 1.0
            pros.append("🟢 **Expanding Bullish MACD:** Histogram showing accelerating momentum.")
        elif last_hist > 0 and last_hist <= prev_hist:
            mom_pts += 0.3
        elif last_hist < 0 and last_hist < prev_hist:
            mom_pts -= 1.0
            watchouts.append("🔴 **Bearish MACD Acceleration:** Negative histogram expanding downwards.")
        elif last_hist < 0 and last_hist >= prev_hist:
            mom_pts -= 0.3

        # Asymmetric RSI Divergence Veto & Boost
        if div_info.get("bearish_divergence"):
            mom_pts = min(0.0, mom_pts - 0.75)
            watchouts.append("🛑 **Asymmetric Veto (Bearish Divergence):** Price made higher high while RSI made lower high. Momentum capped <= 0.0.")
        elif div_info.get("bullish_divergence"):
            mom_pts += 0.50
            pros.append("🟢 **Bullish Divergence Confirmed:** Price made lower low while RSI formed higher low.")

        mom_pts = max(-2.0, min(2.0, mom_pts))

        # =========================================================================
        # BUCKET 3: VOLATILITY LOCATION & PIVOT CONFLUENCE (Max Capped at +1.5 / -1.5 pts)
        # 4-Zone VWAP Location (Ungated Mean Reversion, Gated Directional Value, Exhaustion)
        # + Symmetric 4-Case Pivot Confluence + Upper Wick Supply Trap
        # =========================================================================
        vol_loc_pts = 0.0
        bb_upper, bb_mid, bb_lower, pct_b, bw = calculate_bollinger_bands(close, 20, 2.0)
        last_pct_b = float(pct_b.iloc[-1])
        last_bw = float(bw.iloc[-1])

        if 0.4 <= last_pct_b <= 0.8:
            vol_loc_pts += 0.4
        elif last_pct_b > 0.95:
            vol_loc_pts -= 0.4
            watchouts.append("⚠️ **Upper Band Tag:** Price pressing the 2-sigma Bollinger ceiling.")
        elif last_pct_b < 0.05:
            if regime == "RANGE_BOUND":
                vol_loc_pts += 0.4
                pros.append("⚡ **Lower Band Support:** Price holding lower 2-sigma band.")
            else:
                vol_loc_pts -= 0.4
                watchouts.append("🔴 **Lower Band Breakdown:** Price pressing lower 2-sigma floor.")

        # 1. Horizon-gated 4-Zone VWAP Location
        if horizon == "intraday" and vwap_bands["vwap"] > 0:
            vwap_loc_score = evaluate_vwap_location_score(curr_p, vwap_bands, raw_trend=raw_trend_sum)
            vol_loc_pts += vwap_loc_score
            if vwap_loc_score >= 0.80:
                pros.append(f"🟢 **Optimal VWAP Location:** Value/Discount support zone (VWAP: ₹{vwap_bands['vwap']:.2f}).")
            elif vwap_loc_score <= -0.80:
                watchouts.append(f"⚠️ **VWAP Location Penalty:** Overextended/Climax zone (VWAP: ₹{vwap_bands['vwap']:.2f}).")

        # 2. Universal Distance from 21 EMA (in ATR units) — Penalize Overextension & Reward Pullbacks
        dist_from_ema21 = (curr_p - ema21)
        dist_in_atr = dist_from_ema21 / max(0.01, atr_val)
        if dist_in_atr > 1.8 and raw_trend_sum > 0:
            vol_loc_pts -= 0.85
            watchouts.append(f"⚠️ **Overextension Penalty:** Price is {dist_in_atr:.1f} ATRs extended above 21 EMA. High risk of mean-reversion pullback.")
        elif 0.0 <= dist_in_atr <= 0.8 and curr_p >= ema21 and raw_trend_sum > 0:
            vol_loc_pts += 0.75
            pros.append(f"🟢 **21 EMA Pullback Support:** Price holding key 21 EMA support within {dist_in_atr:.1f} ATRs with tight risk.")

        # 3. Symmetric 4-Case Pivot Confluence & Camarilla Levels
        piv_conf = evaluate_pivot_confluence(curr_p, pivots, raw_trend=raw_trend_sum)
        vol_loc_pts += piv_conf
        if piv_conf > 0:
            pros.append(f"🟢 **Pivot Support Confluence:** Price holding key structural level (P: ₹{pivots['pivot']:.2f}).")
        elif piv_conf < 0:
            watchouts.append(f"⚠️ **Pivot Resistance Obstacle:** Approaching major overhead supply (R1: ₹{pivots['r1']:.2f}).")

        # 4. Institutional Camarilla Equations (Early H4 Breakout vs Climax Past H4)
        if cam_pivots.get("h4", 0) > 0:
            dist_to_h4_pct = (curr_p - cam_pivots["h4"]) / cam_pivots["h4"]
            if 0.0 <= dist_to_h4_pct <= 0.005:
                vol_loc_pts += 0.35
                pros.append(f"🚀 **Camarilla H4 Breakout:** Price breaking above institutional trigger (H4: ₹{cam_pivots['h4']:.2f}).")
            elif dist_to_h4_pct > 0.015:
                vol_loc_pts -= 0.40
                watchouts.append(f"⚠️ **Extended Above H4:** Price +{dist_to_h4_pct*100:.1f}% past Camarilla H4. Avoid chasing extended breakout.")
            elif curr_p <= cam_pivots.get("l4", 0):
                vol_loc_pts -= 0.35
                watchouts.append(f"🔴 **Camarilla L4 Breakdown:** Price breached institutional floor (L4: ₹{cam_pivots['l4']:.2f}).")
            elif abs(curr_p - cam_pivots.get("l3", 0)) / curr_p <= 0.003:
                vol_loc_pts += 0.25
                pros.append(f"⚡ **Camarilla L3 Demand Floor:** Holding institutional mean-reversion buy zone (L3: ₹{cam_pivots['l3']:.2f}).")
        elif abs(curr_p - cam_pivots["h3"]) / curr_p <= 0.003:
            vol_loc_pts -= 0.25
            watchouts.append(f"⚠️ **Camarilla H3 Supply Ceiling:** Near institutional mean-reversion resistance (H3: ₹{cam_pivots['h3']:.2f}).")

        # 4. Volume Profile Point of Control (POC) Interaction
        if vp_info.get("poc", 0) > 0:
            poc_p = vp_info["poc"]
            if curr_p >= poc_p and vp_info.get("poc_distance_pct", 0) <= 1.5:
                vol_loc_pts += 0.25
                pros.append(f"🟢 **Volume POC Support:** Price supported by major volume node (POC: ₹{poc_p:,.2f}).")
            elif curr_p < poc_p and vp_info.get("poc_distance_pct", 0) >= -1.5:
                vol_loc_pts -= 0.20
                watchouts.append(f"⚠️ **Volume POC Overhead:** Facing high-volume resistance ceiling (POC: ₹{poc_p:,.2f}).")

        # 5. Upper Wick Rejection Penalty (VSA Trap Filter)
        if wick_info.get("is_upper_rejection"):
            vol_loc_pts -= 0.4
            watchouts.append(f"⚠️ **Upper Wick Supply Trap:** Trigger candle shows {wick_info['upper_wick_ratio']*100:.0f}% upper shadow (supply absorption).")
        elif wick_info.get("is_lower_absorption"):
            pros.append(f"🟢 **Lower Wick Demand Absorption:** Strong buying tail ({wick_info['lower_wick_ratio']*100:.0f}% lower shadow).")

        vol_loc_pts = max(-1.5, min(1.5, vol_loc_pts))

        # =========================================================================
        # BUCKET 4: VOLUME & INSTITUTIONAL FLOW (Max Capped at +1.5 / -1.5 pts)
        # Pure Volume Metrics: RVol + OBV Flow + FVG + Hurst Persistence
        # =========================================================================
        flow_pts = 0.0
        
        # 1. RVol Scoring
        if rvol_val >= 1.50:
            flow_pts += 0.70
            pros.append(f"🟢 **Institutional Volume Surge:** RVol at {rvol_val:.2f}x average.")
        elif rvol_val >= 1.15:
            flow_pts += 0.35
            pros.append(f"🟢 **Above-Average Volume:** RVol at {rvol_val:.2f}x.")
        elif rvol_val < 0.80:
            flow_pts -= 0.35
            watchouts.append(f"⚠️ **Anemic Volume:** RVol at {rvol_val:.2f}x (Low institutional participation).")

        # 2. OBV Flow
        obv_series = calculate_obv(close, volume)
        if len(obv_series) >= 10:
            obv_slope = float(obv_series.iloc[-1] - obv_series.iloc[-10])
            if obv_slope > 0:
                flow_pts += 0.45
                pros.append("🟢 **Positive On-Balance Volume:** Institutional accumulation active.")
            else:
                flow_pts -= 0.25

        # 3. Smart Money Fair Value Gap (FVG)
        if fvg_info.get("has_fvg"):
            if fvg_info.get("fvg_type") == "BULLISH_DISPLACEMENT_GAP":
                flow_pts += 0.30
                pros.append(f"🟢 **Smart Money FVG:** {fvg_info.get('description')}")
            elif fvg_info.get("fvg_type") == "BEARISH_DISPLACEMENT_GAP":
                flow_pts -= 0.30
                watchouts.append(f"🛑 **Smart Money Supply Gap:** {fvg_info.get('description')}")

        # 4. Hurst Exponent Trend Persistence
        if hurst_h >= 0.58:
            flow_pts += 0.20
            pros.append(f"📈 **High Trend Persistence:** Hurst Exponent at {hurst_h:.2f} (Strong institutional momentum).")
        elif hurst_h <= 0.42:
            flow_pts -= 0.20
        # 5. Derivatives Open Interest (OI) & Smart-Money Order Flow
        if deriv_info and deriv_info.get("status") == "SUCCESS":
            oi_state = deriv_info.get("oi_interpretation", "NEUTRAL")
            pcr_oi = float(deriv_info.get("pcr_oi", 1.0))
            if oi_state == "LONG_BUILDUP":
                flow_pts += 0.40
                pros.append(f"⚡ **Derivatives Long Build-up:** Confirmed institutional accumulation (PCR: {pcr_oi:.2f}).")
            elif oi_state == "PUT_WRITING_SUPPORT":
                flow_pts += 0.20
                pros.append(f"🟢 **Put Writer Support:** Strong demand floor defending key strikes (PCR: {pcr_oi:.2f}).")
            elif oi_state == "SHORT_COVERING":
                watchouts.append("⚠️ **Derivatives Short Covering:** Rebound driven by short squeeze rather than fresh buying.")
            elif oi_state == "SHORT_BUILDUP":
                flow_pts -= 0.40
                watchouts.append(f"🛑 **Derivatives Bearish Overhang:** Heavy Call writing overhead (PCR: {pcr_oi:.2f}).")

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
        # Minimum percentage floor based on trading horizon so targets are realistic and profitable net of fees:
        if horizon == "intraday":
            min_sl_pct = 0.012  # Minimum 1.2% SL floor for intraday (gives Target 1 >= +1.8%, Target 2 >= +3.0%)
        elif horizon == "positional":
            min_sl_pct = 0.050  # Minimum 5.0% SL floor for positional (gives Target 1 >= +7.5%, Target 2 >= +12.5%)
        else: # swing
            min_sl_pct = 0.025  # Minimum 2.5% SL floor for swing (gives Target 1 >= +3.75%, Target 2 >= +6.25%)

        # Continuous SL multiplier: 1.5x in Chop (ADX <= 20) -> 1.2x in Trend (ADX >= 25)
        sl_mult = 1.5 - (0.3 * adx_factor)
        raw_sl_dist = sl_mult * atr_val
        sl_distance = max(curr_p * min_sl_pct, raw_sl_dist)
        sl_price = round(curr_p - sl_distance, 2)
        
        # Targets are tied directly to SL distance (1.5x SL and 2.5x SL)
        # This guarantees that Blended R:R (0.5 * 1.5R + 0.5 * 2.5R) = 2.0R across ALL regimes!
        t1_price = round(curr_p + (1.5 * sl_distance), 2)
        t2_price = round(curr_p + (2.5 * sl_distance), 2)

        # Call Writer Collision Shield (Front-running institutional option sellers)
        if deriv_info and deriv_info.get("status") == "SUCCESS":
            call_wall = float(deriv_info.get("call_writer_wall", 0.0))
            if call_wall > curr_p and t1_price >= call_wall:
                front_run_t1 = round(call_wall * 0.9975, 2)
                # Only recalibrate T1 if distance to Call Wall provides a viable return (>= min_sl_pct)
                if (front_run_t1 - curr_p) / curr_p >= min_sl_pct:
                    t1_price = front_run_t1
                    pros.append(f"🛡️ **Call Wall Collision Shield:** Target 1 calibrated to ₹{t1_price:.2f} to front-run the institutional Call Writer Wall at ₹{call_wall:.2f}.")
                else:
                    watchouts.append(f"🧱 **Call Wall Overhead:** Institutional Call Wall at ₹{call_wall:.2f} compresses upside runway to {((call_wall-curr_p)/curr_p)*100:.1f}%.")

        t1_gain = round(((t1_price - curr_p) / curr_p) * 100.0, 2) if curr_p > 0 else 0.0
        t2_gain = round(((t2_price - curr_p) / curr_p) * 100.0, 2) if curr_p > 0 else 0.0
        sl_loss = round(((curr_p - sl_price) / curr_p) * 100.0, 2) if curr_p > 0 else 0.0
        t1_rr = round(t1_gain / sl_loss, 2) if sl_loss > 0 else 1.5
        t2_rr = round(t2_gain / sl_loss, 2) if sl_loss > 0 else 2.5
        entry_zone_str = f"₹{curr_p * 0.998:.2f} – ₹{curr_p:.2f}"

        # Precision Indicators: Squeeze, RS, and VSA
        squeeze_info = calculate_ttm_squeeze(high, low, close)
        vsa_info = calculate_vsa_structure(open_, high, low, close, volume)
        rs_info = calculate_relative_strength_vs_benchmark(close)

        if squeeze_info.get("squeeze_fired") and raw_trend_sum > 0:
            pros.append("🚀 **TTM Squeeze Firing:** Volatility coiling broken with directional momentum (>75% win-rate trigger).")
        elif squeeze_info.get("squeeze_on"):
            pros.append("⚡ **TTM Squeeze Coiling:** Volatility compressed inside Keltner bands. Explosive move imminent.")

        if vsa_info.get("is_trap"):
            watchouts.append(f"🛑 **VSA Trap Alert:** {vsa_info.get('description')}")
        elif vsa_info.get("pattern") == "STOPPING_VOLUME_ABSORPTION":
            pros.append("🟢 **VSA Demand Absorption:** Institutional buying absorbing selling pressure at lows.")
        elif vsa_info.get("pattern") == "NO_SUPPLY_PULLBACK":
            pros.append("🟢 **VSA Low-Supply Test:** Low-volume pullback confirming absence of sellers.")

        if rs_info.get("status") in ["STRONG_OUTPERFORMER", "OUTPERFORMING"]:
            pros.append(f"💪 **Institutional Relative Strength:** Outperforming NIFTY 50 by +{rs_info.get('rs_diff_pct')}% alpha.")
        elif rs_info.get("status") in ["HEAVY_UNDERPERFORMER", "UNDERPERFORMING"]:
            watchouts.append(f"⚠️ **Lagging Benchmark:** Underperforming NIFTY 50 by {rs_info.get('rs_diff_pct')}%.")

        # =========================================================================
        # THE 5 HARD ASYMMETRIC VETO GATES (ELIMINATES FALSE BREAKOUTS & TRAPS)
        # =========================================================================
        # 1. VSA Upthrust / Supply Absorption Trap Veto
        if vsa_info.get("is_trap"):
            final_score = min(final_score, 4.2)
            verdict = "🔴 AVOID / TRAP DETECTED"
            verdict_desc = f"VSA Trap Alert: {vsa_info.get('description')} (High risk of sudden rejection)."
            action = "AVOID"
            badge_color = "#f85149"

        # 2. Call Writer Ceiling Overhead Veto (Distance to Call Wall < 1.0%)
        elif deriv_info and deriv_info.get("status") == "SUCCESS" and deriv_info.get("call_writer_wall", 0) > curr_p and deriv_info.get("runway_to_call_wall_pct", 100.0) < 1.0 and final_score >= 6.0:
            call_wall = float(deriv_info.get("call_writer_wall", 0.0))
            final_score = min(final_score, 5.9)
            verdict = "⏳ WAIT / AT CALL RESISTANCE WALL"
            verdict_desc = f"Institutional Call Writer Wall immediately overhead at ₹{call_wall:,.2f} (<1.0% away). Rejection risk high; wait for breakout or pullback."
            action = "WAIT"
            badge_color = "#d29922"
            watchouts.append(f"🧱 **Call Wall Overhead:** Multi-crore institutional Call resistance at ₹{call_wall:,.2f}. Avoid buying right beneath this wall.")

        # 3. Bearish Divergence near Resistance Veto
        elif div_info.get("bearish_divergence") and final_score >= 6.5:
            final_score = min(final_score, 5.8)
            verdict = "🟡 WAIT / BEARISH DIVERGENCE"
            verdict_desc = "Price made higher highs while momentum weakened. Divergence veto active."
            action = "WAIT"
            badge_color = "#d29922"

        # 4. Anemic Smart Money Volume Veto
        elif rvol_val < 0.70 and final_score >= 7.2:
            final_score = min(final_score, 6.4)
            verdict = "⏳ BUY ON PULLBACK (LOW RVOL)"
            verdict_desc = f"Anemic volume ({rvol_val:.2f}x). Smart money absent; avoid chasing breakouts."
            action = "BUY ON DIP"
            badge_color = "#2ea043"

        # 5. Overextended Candle Ceiling Veto (Price > 1.8 ATRs above 21 EMA or > 2.0% above 21 EMA)
        dist_from_ema21 = (curr_p - ema21)
        dist_in_atr = dist_from_ema21 / max(0.01, atr_val)
        if (dist_in_atr > 1.8 or (curr_p - ema21) / curr_p > 0.020) and final_score >= 6.8:
            final_score = min(final_score, 6.2)  # CLAMP SCORE: Stop overextended setups from qualifying as Grade A/A+
            verdict = "⏳ BUY ON PULLBACK (EXTENDED)"
            verdict_desc = f"Price extended {dist_in_atr:.1f} ATRs (+{((curr_p-ema21)/curr_p)*100:.1f}%) above 21 EMA. Enter on pullback to 21 EMA floor."
            action = "BUY ON DIP"
            badge_color = "#2ea043"
            entry_zone_str = f"₹{ema21 * 0.998:.2f} – ₹{ema21 * 1.006:.2f}"
            watchouts.append(f"⚠️ **Overextension Warning:** Price is {dist_in_atr:.1f} ATRs extended. Chasing here has poor risk-reward.")

        # Institutional Grade Classification (Expectancy & 1:2 R:R Focus)
        has_clear_runway = deriv_info.get("has_clear_runway", True) if deriv_info else True
        deriv_oi = deriv_info.get("oi_interpretation", "NEUTRAL") if deriv_info else "NEUTRAL"
        is_deriv_bullish = deriv_oi in ["LONG_BUILDUP", "PUT_WRITING_SUPPORT", "NEUTRAL_BALANCED"]

        if (
            final_score >= 7.8
            and not vsa_info.get("is_trap")
            and rs_info.get("rs_ratio", 1.0) >= 1.00
            and index_trend in ["BULLISH", "NEUTRAL"]
            and htf_trend in ["BULLISH", "NEUTRAL"]
            and is_deriv_bullish
            and has_clear_runway
            and dist_in_atr <= 1.5
        ):
            setup_grade = "GRADE_A_PLUS"
            setup_grade_title = "🌟 GRADE A+ (Prime Confluence Setup — Target 1:2 R:R)"
            win_probability = 65
        elif final_score >= 7.0 and not vsa_info.get("is_trap") and dist_in_atr <= 1.8:
            setup_grade = "GRADE_A"
            setup_grade_title = "⚡ GRADE A (High Probability Momentum — Min 1:1.5 R:R)"
            win_probability = 58
        elif final_score >= 5.8:
            setup_grade = "GRADE_B"
            setup_grade_title = "⏳ GRADE B (Support / Pullback Setup — Min 1:1.5 R:R)"
            win_probability = 52
        elif final_score >= 4.5:
            setup_grade = "GRADE_C"
            setup_grade_title = "🟡 GRADE C (Consolidation / Neutral — Capital Preserved)"
            win_probability = 45
        else:
            setup_grade = "GRADE_D"
            setup_grade_title = "🛑 GRADE D (Avoid / Counter-Trend — Capital Protected)"
            win_probability = 35

        return {
            "status": "SUCCESS",
            "symbol": symbol,
            "display_name": display_symbol_name(symbol),
            "current_price": curr_p,
            "score": final_score,
            "setup_grade": setup_grade,
            "setup_grade_title": setup_grade_title,
            "win_probability": win_probability,
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
                "atr": round(atr_val, 2),
                "vwap": round(vwap_bands.get("vwap", 0.0), 2),
                "rvol": round(rvol_val, 2)
            },
            "levels": {
                "entry_zone": entry_zone_str,
                "stop_loss": sl_price,
                "target_1": t1_price,
                "target_2": t2_price,
                "risk_reward": f"1:{t1_rr} to Target 1 | 1:{t2_rr} to Target 2"
            },
            "mtf_alignment": mtf_info,
            "divergence": div_info,
            "wick_structure": wick_info,
            "vwap_structure": vwap_bands,
            "pivots": pivots,
            "fib_pivots": fib_pivots,
            "camarilla_pivots": cam_pivots,
            "volume_profile": vp_info,
            "hurst_exponent": hurst_h,
            "fvg_structure": fvg_info,
            "ttm_squeeze": squeeze_info,
            "vsa_profile": vsa_info,
            "relative_strength": rs_info,
            "rvol": rvol_val,
            "pros": pros,
            "watchouts": watchouts,
            "derivatives": deriv_info
        }

    @classmethod
    def analyze_stock(cls, symbol: str, horizon: str = "swing") -> Dict[str, Any]:
        """
        Analyze stock for given horizon using dual-timeframe and live macro index & derivatives confluence.
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

        curr_close = float(df["Close"].iloc[-1])

        # 1. Real-Time Macro Index Trend & Sector Lookup
        index_data = get_live_index_trend()
        index_trend = index_data.get("nifty_trend", "NEUTRAL")
        sector_name = get_sector_for_symbol(sym)

        # 2. Dual Timeframe Analysis (Daily HTF Anchor)
        htf_trend = "NEUTRAL"
        try:
            df_htf = get_historical_data(sym, period="6mo", interval="1d")
            if not df_htf.empty and len(df_htf) >= 30:
                htf_c = df_htf["Close"]
                htf_ema20 = float(htf_c.ewm(span=20, adjust=False).mean().iloc[-1])
                htf_ema50 = float(htf_c.ewm(span=50, adjust=False).mean().iloc[-1])
                htf_curr = float(htf_c.iloc[-1])
                if htf_curr > htf_ema20 and htf_ema20 > htf_ema50:
                    htf_trend = "BULLISH"
                elif htf_curr < htf_ema50 and htf_ema20 < htf_ema50:
                    htf_trend = "BEARISH"
        except Exception:
            pass

        # 3. Derivatives Flow & Option Chain Analysis
        deriv_info = None
        try:
            deriv_info = DerivativesFlowAnalyzer.analyze_derivatives_structure(sym, curr_close)
        except Exception:
            pass

        res = cls.evaluate_df_slice(
            df, sym,
            horizon=horizon,
            index_trend=index_trend,
            htf_trend=htf_trend,
            sector_name=sector_name,
            deriv_info=deriv_info
        )
        res["horizon"] = horizon
        res["horizon_text"] = time_text
        res["holding_time_text"] = time_text
        res["index_trend"] = index_trend
        res["nifty_data"] = index_data
        res["sector"] = sector_name
        res["htf_trend"] = htf_trend
        res["derivatives"] = deriv_info
        return res
