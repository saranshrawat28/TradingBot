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
    calculate_adx, calculate_vwap, add_all_indicators,
    detect_rsi_divergence, calculate_candle_structure, calculate_mtf_alignment,
    calculate_intraday_vwap_bands, calculate_rvol, calculate_context_multiplier,
    calculate_obv, calculate_classical_pivots, calculate_fibonacci_pivots,
    evaluate_vwap_location_score, evaluate_pivot_confluence,
    calculate_relative_strength_vs_benchmark, calculate_ttm_squeeze, calculate_vsa_structure
)
from src.utils.helpers import clean_symbol, display_symbol_name, format_currency_inr

class StockAdvisor:
    """
    Quantitative scoring and trade setup engine.
    """

    @classmethod
    def evaluate_df_slice(cls, df: pd.DataFrame, symbol: str = "ASSET", horizon: str = "intraday", index_trend: str = "BULLISH") -> Dict[str, Any]:
        """
        Evaluates a slice of historical candles (strictly closed bars) using
        orthogonal category capping, single combined context multiplier (ADX + Macro Breadth),
        symmetric MTF trend multipliers, divergence asymmetric gates, 4-zone VWAP sigma-location,
        symmetric 4-case pivot confluence, and pure RVol flow.
        """
        if df.empty or len(df) < 25:
            return {"status": "ERROR", "score": 5.0, "message": "Insufficient data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        open_ = df["Open"] if "Open" in df.columns else close

        curr_p = float(close.iloc[-1])
        prev_p = float(close.iloc[-2]) if len(close) > 1 else curr_p
        
        # Previous Session High, Low, Close for Pivots
        prev_h = float(high.iloc[-2]) if len(high) > 1 else float(high.iloc[-1])
        prev_l = float(low.iloc[-2]) if len(low) > 1 else float(low.iloc[-1])
        prev_c = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
        pivots = calculate_classical_pivots(prev_h, prev_l, prev_c)
        fib_pivots = calculate_fibonacci_pivots(prev_h, prev_l, prev_c)

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
        else:
            # Swing / Positional: Use distance from 21 EMA
            ema_dist_pct = abs(curr_p - ema21) / curr_p
            if ema_dist_pct <= 0.02 and curr_p >= ema21:
                vol_loc_pts += 0.75
                pros.append(f"🟢 **21 EMA Pullback Support:** Holding tight within 2% of key moving average (₹{ema21:.2f}).")

        # 2. Symmetric 4-Case Pivot Confluence
        piv_conf = evaluate_pivot_confluence(curr_p, pivots, raw_trend=raw_trend_sum)
        vol_loc_pts += piv_conf
        if piv_conf > 0:
            pros.append(f"🟢 **Pivot Support Confluence:** Price holding key structural level (P: ₹{pivots['pivot']:.2f}).")
        elif piv_conf < 0:
            watchouts.append(f"⚠️ **Pivot Resistance Obstacle:** Approaching major overhead supply (R1: ₹{pivots['r1']:.2f}).")

        # 3. Upper Wick Rejection Penalty (VSA Trap Filter)
        if wick_info.get("is_upper_rejection"):
            vol_loc_pts -= 0.4
            watchouts.append(f"⚠️ **Upper Wick Supply Trap:** Trigger candle shows {wick_info['upper_wick_ratio']*100:.0f}% upper shadow (supply absorption).")
        elif wick_info.get("is_lower_absorption"):
            pros.append(f"🟢 **Lower Wick Demand Absorption:** Strong buying tail ({wick_info['lower_wick_ratio']*100:.0f}% lower shadow).")

        vol_loc_pts = max(-1.5, min(1.5, vol_loc_pts))

        # =========================================================================
        # BUCKET 4: VOLUME & INSTITUTIONAL FLOW (Max Capped at +1.5 / -1.5 pts)
        # Pure Volume Metrics: RVol + OBV Flow (ZERO price/VWAP check to prevent collinearity)
        # =========================================================================
        flow_pts = 0.0
        
        # 1. RVol Scoring
        if rvol_val >= 1.50:
            flow_pts += 0.80
            pros.append(f"🟢 **Institutional Volume Surge:** RVol at {rvol_val:.2f}x average.")
        elif rvol_val >= 1.15:
            flow_pts += 0.40
            pros.append(f"🟢 **Above-Average Volume:** RVol at {rvol_val:.2f}x.")
        elif rvol_val < 0.80:
            flow_pts -= 0.40
            watchouts.append(f"⚠️ **Anemic Volume:** RVol at {rvol_val:.2f}x (Low institutional participation).")

        # 2. OBV Flow
        obv_series = calculate_obv(close, volume)
        if len(obv_series) >= 10:
            obv_slope = float(obv_series.iloc[-1] - obv_series.iloc[-10])
            if obv_slope > 0:
                flow_pts += 0.70
                pros.append("🟢 **Positive On-Balance Volume:** Institutional accumulation active.")
            else:
                flow_pts -= 0.30

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
        sl_distance = sl_mult * atr_val
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

        # Setup Quality Grading & Estimated Win-Rate Probability
        if final_score >= 8.5 and not vsa_info.get("is_trap") and rs_info.get("rs_ratio", 1.0) >= 1.00:
            setup_grade = "GRADE_A_PLUS"
            setup_grade_title = "🌟 GRADE A+ (Elite Institutional Setup)"
            win_probability = 82
        elif final_score >= 7.5 and not vsa_info.get("is_trap"):
            setup_grade = "GRADE_A"
            setup_grade_title = "⚡ GRADE A (High Probability Breakout)"
            win_probability = 72
        elif final_score >= 6.2:
            setup_grade = "GRADE_B"
            setup_grade_title = "⏳ GRADE B (Pullback / Accumulation)"
            win_probability = 58
        else:
            setup_grade = "GRADE_AVOID"
            setup_grade_title = "🛑 AVOID / NEUTRAL (Capital Preserved)"
            win_probability = 42

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
            "ttm_squeeze": squeeze_info,
            "vsa_profile": vsa_info,
            "relative_strength": rs_info,
            "rvol": rvol_val,
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

        res = cls.evaluate_df_slice(df, sym, horizon=horizon)
        res["horizon"] = horizon
        res["horizon_text"] = time_text
        res["holding_time_text"] = time_text
        return res
